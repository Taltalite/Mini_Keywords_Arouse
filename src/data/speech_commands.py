from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import Dataset

try:
    import torchaudio
except ImportError as exc:  # pragma: no cover - exercised before dependencies are installed
    raise ImportError(
        "torchaudio is required. Install dependencies with: uv pip install -r requirements.txt"
    ) from exc


TARGET_WORDS: tuple[str, ...] = (
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
)
ALL_LABELS: tuple[str, ...] = (*TARGET_WORDS, "unknown", "silence")
LABEL_TO_INDEX: dict[str, int] = {label: idx for idx, label in enumerate(ALL_LABELS)}


def label_to_index(label: str) -> int:
    mapped = label if label in TARGET_WORDS else "unknown"
    return LABEL_TO_INDEX[mapped]


def _fix_length(
    waveform: torch.Tensor,
    num_samples: int,
    training: bool,
    rng: random.Random | None = None,
) -> torch.Tensor:
    current = waveform.shape[-1]
    if current == num_samples:
        return waveform
    if current < num_samples:
        pad = num_samples - current
        return torch.nn.functional.pad(waveform, (0, pad))

    max_start = current - num_samples
    start = (rng or random).randint(0, max_start) if training else max_start // 2
    return waveform[..., start : start + num_samples]


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


class SpeechCommandsKWS(Dataset):
    """12-class Speech Commands wrapper for keyword spotting.

    Labels are the 10 target words plus ``unknown`` and synthetic ``silence``.
    Audio is returned as ``[1, num_samples]`` float tensor at ``sample_rate``.
    """

    def __init__(
        self,
        data_root: str | Path = "data/SpeechCommands",
        subset: str = "training",
        sample_rate: int = 16_000,
        num_samples: int = 16_000,
        limit: int | None = None,
        download: bool = True,
        silence_ratio: float = 0.05,
        silence_gain_min: float = 0.0,
        silence_gain_max: float = 0.001,
        seed: int = 1337,
    ) -> None:
        if subset not in {"training", "validation", "testing"}:
            raise ValueError(f"Unsupported subset: {subset}")

        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.subset = subset
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.training = subset == "training"
        self.rng = random.Random(seed)
        self.silence_gain_min = silence_gain_min
        self.silence_gain_max = silence_gain_max
        if self.silence_gain_min < 0 or self.silence_gain_max < self.silence_gain_min:
            raise ValueError("silence_gain_min/max must satisfy 0 <= min <= max")

        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(self.data_root),
            url="speech_commands_v0.02",
            download=download,
            subset=subset,
        )

        indices = list(range(len(self.dataset)))
        self.rng.shuffle(indices)
        if limit is not None:
            if limit <= 0:
                raise ValueError("--limit must be positive when provided")
            indices = indices[: min(limit, len(indices))]
        self.indices = indices

        silence_count = max(1, int(len(self.indices) * silence_ratio)) if self.indices else 0
        self.silence_items = list(range(silence_count))
        self.background_noise_paths = self._find_background_noise_paths()
        self._background_cache: dict[Path, tuple[int, torch.Tensor]] = {}

    @property
    def labels(self) -> tuple[str, ...]:
        return ALL_LABELS

    def __len__(self) -> int:
        return len(self.indices) + len(self.silence_items)

    def target_indices(self) -> list[int]:
        targets: list[int] = []
        for dataset_index in self.indices:
            _, _, label, _, _ = self.dataset.get_metadata(dataset_index)
            targets.append(label_to_index(str(label)))
        targets.extend([LABEL_TO_INDEX["silence"]] * len(self.silence_items))
        return targets

    def label_counts(self) -> dict[str, int]:
        counts = {label: 0 for label in ALL_LABELS}
        for target in self.target_indices():
            counts[ALL_LABELS[target]] += 1
        return counts

    def silence_strategy(self) -> dict[str, object]:
        return {
            "source": "background_noise" if self.background_noise_paths else "synthetic",
            "background_noise_files": [str(path) for path in self.background_noise_paths],
            "fallback": "50% zeros, 50% low-amplitude random noise",
            "gain_min": self.silence_gain_min,
            "gain_max": self.silence_gain_max,
            "count": len(self.silence_items),
        }

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        if idx >= len(self.indices):
            waveform = self._make_silence()
            return waveform, LABEL_TO_INDEX["silence"]

        waveform, source_sr, label = self._load_public_item(self.indices[idx])
        waveform = _to_mono(waveform).float()
        if source_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, source_sr, self.sample_rate)
        waveform = _fix_length(waveform, self.num_samples, self.training, self.rng)
        return waveform, label_to_index(label)

    def _load_public_item(self, dataset_index: int) -> tuple[torch.Tensor, int, str]:
        """Load one Speech Commands item without touching torchaudio private fields."""
        try:
            waveform, source_sr, label, _, _ = self.dataset[dataset_index]
            return waveform, int(source_sr), str(label)
        except RuntimeError as exc:
            message = str(exc)
            if "torchcodec" not in message and "AudioDecoder" not in message:
                raise

            audio_path, source_sr, label, _, _ = self.dataset.get_metadata(dataset_index)
            audio_path = self._resolve_metadata_audio_path(audio_path)
            fallback_sr, audio = wavfile.read(audio_path)
            if int(source_sr) != int(fallback_sr):
                raise RuntimeError(
                    f"Speech Commands metadata sample rate {source_sr} does not match "
                    f"decoded WAV sample rate {fallback_sr} for {audio_path}"
                ) from exc
            return self._wav_to_tensor(audio), int(fallback_sr), str(label)

    def _resolve_metadata_audio_path(self, audio_path: str | Path) -> Path:
        path = Path(audio_path)
        if path.exists():
            return path

        candidates = (
            self.data_root / "SpeechCommands" / path,
            self.data_root / path,
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not resolve Speech Commands audio path: {audio_path}")

    def _make_silence(self) -> torch.Tensor:
        if self.background_noise_paths:
            return self._make_background_silence()
        if self.rng.random() < 0.5:
            noise = torch.randn(1, self.num_samples) * self._sample_silence_gain()
            return noise.float()
        return torch.zeros(1, self.num_samples, dtype=torch.float32)

    def _sample_silence_gain(self) -> float:
        return self.rng.uniform(self.silence_gain_min, self.silence_gain_max)

    def _make_background_silence(self) -> torch.Tensor:
        path = self.rng.choice(self.background_noise_paths)
        source_sr, waveform = self._load_background_noise(path)
        waveform = _to_mono(waveform).float()
        if source_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, source_sr, self.sample_rate)
        waveform = _fix_length(waveform, self.num_samples, training=True, rng=self.rng)
        return (waveform * self._sample_silence_gain()).float()

    def _load_background_noise(self, path: Path) -> tuple[int, torch.Tensor]:
        if path not in self._background_cache:
            source_sr, audio = wavfile.read(path)
            self._background_cache[path] = (int(source_sr), self._wav_to_tensor(audio))
        return self._background_cache[path]

    def _find_background_noise_paths(self) -> list[Path]:
        candidates = (
            self.data_root / "SpeechCommands" / "speech_commands_v0.02" / "_background_noise_",
            self.data_root / "SpeechCommands" / "SpeechCommands" / "speech_commands_v0.02" / "_background_noise_",
            self.data_root / "speech_commands_v0.02" / "_background_noise_",
        )
        paths: list[Path] = []
        for directory in candidates:
            if directory.exists():
                paths.extend(sorted(directory.glob("*.wav")))
        return paths

    @staticmethod
    def _wav_to_tensor(audio: np.ndarray) -> torch.Tensor:
        if audio.ndim == 1:
            audio = audio[None, :]
        else:
            audio = audio.T

        if np.issubdtype(audio.dtype, np.integer):
            scale = float(np.iinfo(audio.dtype).max)
            audio = audio.astype(np.float32) / scale
        else:
            audio = audio.astype(np.float32)
        return torch.from_numpy(audio)


def summarize_labels(dataset: Iterable[tuple[torch.Tensor, int]]) -> dict[str, int]:
    counts = {label: 0 for label in ALL_LABELS}
    if isinstance(dataset, SpeechCommandsKWS):
        return dataset.label_counts()
    for _, target in dataset:
        counts[ALL_LABELS[int(target)]] += 1
    return counts
