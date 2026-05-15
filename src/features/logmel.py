from __future__ import annotations

import torch
from torch import nn

try:
    import torchaudio
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torchaudio is required. Install dependencies with: uv pip install -r requirements.txt"
    ) from exc


class LogMelExtractor(nn.Module):
    """Log-Mel feature extractor returning [B, 1, n_mels, time]."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 40,
        f_min: float = 20.0,
        f_max: float = 7_600.0,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
            center=True,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.dim() != 3:
            raise ValueError(f"Expected waveform shape [B, 1, T] or [B, T], got {waveform.shape}")

        features = torch.log(self.mel(waveform).clamp_min(1e-6))
        if self.normalize:
            mean = features.mean(dim=(-2, -1), keepdim=True)
            std = features.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
            features = (features - mean) / std
        return features
