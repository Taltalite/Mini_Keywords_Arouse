from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

from src.data.speech_commands import LABEL_TO_INDEX, TARGET_WORDS, SpeechCommandsKWS
from src.features.logmel import LogMelExtractor
from src.models import build_model, normalize_model_config


SAMPLE_RATE = 16_000
WINDOW_SEC = 1.0
HOP_SEC = 0.1
STREAM_DURATION_SEC = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate streaming keyword spotting.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path.")
    parser.add_argument("--target", required=True, choices=TARGET_WORDS, help="Target keyword.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Detection threshold.")
    parser.add_argument("--window-sec", type=float, default=WINDOW_SEC, help="Window size in seconds.")
    parser.add_argument("--hop-sec", type=float, default=HOP_SEC, help="Hop size in seconds.")
    parser.add_argument("--smooth-alpha", type=float, default=0.3, help="EMA smoothing alpha.")
    parser.add_argument("--trigger-consecutive", type=int, default=3, help="Consecutive windows above threshold to trigger.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for demo audio construction.")
    parser.add_argument("--output", default="outputs/stream_demo.json", help="Output JSON path.")
    parser.add_argument("--data-root", default=None, help="Override dataset root.")
    return parser.parse_args()


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_demo_audio(
    target_word: str,
    data_cfg: dict[str, Any],
    seed: int,
    duration_sec: float = STREAM_DURATION_SEC,
) -> tuple[torch.Tensor, float]:
    """Build a synthetic continuous audio stream with one target keyword in the middle."""
    rng = random.Random(seed)
    sample_rate = int(data_cfg.get("sample_rate", SAMPLE_RATE))
    num_samples = int(data_cfg.get("num_samples", sample_rate))
    total_samples = int(duration_sec * sample_rate)

    # Try to fetch a real target sample from the testing subset.
    dataset = SpeechCommandsKWS(
        data_root=data_cfg.get("data_root", "data/SpeechCommands"),
        subset="testing",
        sample_rate=sample_rate,
        num_samples=num_samples,
        download=False,
        seed=seed,
    )
    target_idx = LABEL_TO_INDEX[target_word]
    all_targets = dataset.target_indices()
    target_indices = [i for i in range(len(dataset.indices)) if all_targets[i] == target_idx]

    stream = torch.zeros(1, total_samples, dtype=torch.float32)
    keyword_start = total_samples // 2 - num_samples // 2

    if target_indices:
        chosen = rng.choice(target_indices)
        keyword_waveform, _ = dataset[chosen]
        keyword_length = keyword_waveform.shape[-1]
        if keyword_length > num_samples:
            keyword_waveform = keyword_waveform[..., :num_samples]
            keyword_length = num_samples
        end = min(keyword_start + keyword_length, total_samples)
        stream[..., keyword_start:end] = keyword_waveform[..., : end - keyword_start]
    else:
        # Fallback: insert a short synthetic pulse if no real sample is available.
        pulse = torch.sin(2 * np.pi * 440.0 * torch.arange(num_samples) / sample_rate).unsqueeze(0)
        end = min(keyword_start + num_samples, total_samples)
        stream[..., keyword_start:end] = pulse[..., : end - keyword_start]

    return stream, float(keyword_start) / sample_rate


def run_streaming(
    model: torch.nn.Module,
    feature_extractor: LogMelExtractor,
    audio: torch.Tensor,
    target_idx: int,
    window_sec: float,
    hop_sec: float,
    threshold: float,
    smooth_alpha: float,
    trigger_consecutive: int,
) -> dict[str, Any]:
    sample_rate = SAMPLE_RATE
    window_samples = int(window_sec * sample_rate)
    hop_samples = int(hop_sec * sample_rate)
    total_samples = audio.shape[-1]

    model.eval()
    timeline: list[dict[str, Any]] = []
    smoothed_prob = 0.0
    consecutive = 0
    triggered = False
    trigger_time_sec: float | None = None

    with torch.no_grad():
        start = 0
        while start + window_samples <= total_samples:
            window = audio[..., start : start + window_samples]
            features = feature_extractor(window)  # [1, 1, n_mels, time]
            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            raw_prob = float(probs[0, target_idx].item())

            smoothed_prob = smooth_alpha * raw_prob + (1 - smooth_alpha) * smoothed_prob
            above = smoothed_prob >= threshold
            consecutive = consecutive + 1 if above else 0

            time_sec = start / sample_rate
            if consecutive >= trigger_consecutive and not triggered:
                triggered = True
                trigger_time_sec = time_sec

            timeline.append(
                {
                    "time_sec": round(time_sec, 3),
                    "raw_probability": round(raw_prob, 6),
                    "smoothed_probability": round(smoothed_prob, 6),
                    "above_threshold": above,
                    "triggered": triggered,
                }
            )
            start += hop_samples

    return {
        "target": TARGET_WORDS[target_idx],
        "threshold": threshold,
        "triggered": triggered,
        "trigger_time_sec": trigger_time_sec,
        "window_sec": window_sec,
        "hop_sec": hop_sec,
        "smooth_alpha": smooth_alpha,
        "trigger_consecutive": trigger_consecutive,
        "timeline": timeline,
    }


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.ckpt)
    data_cfg = dict(checkpoint.get("data_config", {}))
    if args.data_root is not None:
        data_cfg["data_root"] = args.data_root
    feature_cfg = dict(checkpoint.get("feature_config", {}))
    model_cfg = normalize_model_config(dict(checkpoint.get("model_config", {})), feature_cfg)

    device = torch.device("cpu")
    feature_extractor = LogMelExtractor(**feature_cfg).to(device)
    model = build_model(model_cfg, feature_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    audio, keyword_time_sec = build_demo_audio(args.target, data_cfg, args.seed)
    target_idx = LABEL_TO_INDEX[args.target]

    result = run_streaming(
        model,
        feature_extractor,
        audio,
        target_idx,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        threshold=args.threshold,
        smooth_alpha=args.smooth_alpha,
        trigger_consecutive=args.trigger_consecutive,
    )
    result["keyword_insertion_time_sec"] = round(keyword_time_sec, 3)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(f"target: {args.target}")
    print(f"threshold: {args.threshold}")
    print(f"triggered: {result['triggered']}")
    print(f"trigger_time_sec: {result['trigger_time_sec']}")
    print(f"keyword_insertion_time_sec: {result['keyword_insertion_time_sec']}")
    print(f"timeline windows: {len(result['timeline'])}")
    print(f"saved stream demo: {output_path}")


if __name__ == "__main__":
    main()
