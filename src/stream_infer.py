from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch

from src.data.speech_commands import SpeechCommandsKWS
from src.features.logmel import LogMelExtractor
from src.models import build_model, normalize_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming keyword spotting simulation.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path.")
    parser.add_argument("--target", required=True, help="Target keyword to detect.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Detection threshold.")
    parser.add_argument("--window-sec", type=float, default=1.0, help="Sliding window size in seconds.")
    parser.add_argument("--hop-sec", type=float, default=0.1, help="Hop size in seconds.")
    parser.add_argument("--smooth-window", type=int, default=5, help="Moving average window for confidence smoothing.")
    parser.add_argument("--trigger-consecutive", type=int, default=3, help="Required consecutive windows above threshold to trigger.")
    parser.add_argument("--data-root", default="data/SpeechCommands", help="Dataset root.")
    parser.add_argument("--output", default="outputs/stream_demo.json", help="Output JSON path.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for sample selection.")
    return parser.parse_args()


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_streaming_buffer(
    dataset: SpeechCommandsKWS,
    target_label: str,
    pre_silence_sec: float = 2.0,
    post_silence_sec: float = 2.0,
    sample_rate: int = 16000,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, float]:
    """Build a synthetic continuous audio buffer: [pre-silence][keyword][post-silence]."""
    rng = rng or random.Random(0)
    target_indices = []
    real_count = len(dataset.indices)
    for idx in range(real_count):
        _, label = dataset[idx]
        if dataset.labels[label] == target_label:
            target_indices.append(idx)

    if not target_indices:
        raise ValueError(f"No '{target_label}' samples found in dataset.")

    chosen_idx = rng.choice(target_indices)
    keyword_waveform, _ = dataset[chosen_idx]  # [1, num_samples]

    num_pre = int(pre_silence_sec * sample_rate)
    num_post = int(post_silence_sec * sample_rate)
    pre_silence = torch.zeros(1, num_pre, dtype=torch.float32)
    post_silence = torch.zeros(1, num_post, dtype=torch.float32)

    buffer = torch.cat([pre_silence, keyword_waveform, post_silence], dim=1)
    keyword_start_sec = pre_silence_sec
    return buffer, keyword_start_sec


def run_streaming_inference(
    model: torch.nn.Module,
    feature_extractor: LogMelExtractor,
    waveform_buffer: torch.Tensor,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
    smooth_window: int,
    trigger_consecutive: int,
    target_index: int,
    threshold: float,
    device: torch.device,
) -> tuple[bool, float | None, list[dict[str, Any]]]:
    num_window_samples = int(window_sec * sample_rate)
    num_hop_samples = int(hop_sec * sample_rate)
    total_samples = waveform_buffer.shape[-1]

    # Extract windows
    windows: list[torch.Tensor] = []
    start = 0
    while start + num_window_samples <= total_samples:
        window = waveform_buffer[:, start : start + num_window_samples]
        windows.append(window)
        start += num_hop_samples

    # Run inference on each window
    model.eval()
    raw_probs: list[float] = []
    with torch.no_grad():
        for window in windows:
            window = window.to(device).unsqueeze(0)  # [1, 1, T]
            features = feature_extractor(window)  # [1, 1, n_mels, time]
            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            raw_probs.append(probs[0, target_index].item())

    # Confidence smoothing: moving average
    smoothed_probs: list[float] = []
    for i in range(len(raw_probs)):
        vals = raw_probs[max(0, i - smooth_window + 1) : i + 1]
        smoothed_probs.append(sum(vals) / len(vals))

    # Trigger detection
    triggered = False
    trigger_time_sec: float | None = None
    consecutive = 0

    timeline: list[dict[str, Any]] = []
    for i, (raw, smooth) in enumerate(zip(raw_probs, smoothed_probs)):
        time_sec = round(i * hop_sec, 2)
        entry = {
            "time_sec": time_sec,
            "raw_probability": round(raw, 4),
            "smoothed_probability": round(smooth, 4),
        }
        timeline.append(entry)

        if smooth >= threshold:
            consecutive += 1
            if consecutive >= trigger_consecutive and not triggered:
                triggered = True
                trigger_time_sec = time_sec
        else:
            consecutive = 0

    return triggered, trigger_time_sec, timeline


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    checkpoint = load_checkpoint(args.ckpt)
    model_cfg = dict(checkpoint.get("model_config", {}))
    feature_cfg = dict(checkpoint.get("feature_config", {}))
    data_cfg = dict(checkpoint.get("data_config", {}))
    model_cfg.setdefault("name", "small_cnn")
    model_cfg = normalize_model_config(model_cfg, feature_cfg)

    device = torch.device("cpu")
    feature_extractor = LogMelExtractor(**feature_cfg).to(device)
    model = build_model(model_cfg, feature_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels = list(checkpoint.get("labels", []))
    if args.target not in labels:
        raise ValueError(f"Target '{args.target}' not in model labels: {labels}")
    target_index = labels.index(args.target)

    sample_rate = int(data_cfg.get("sample_rate", 16000))

    dataset = SpeechCommandsKWS(
        data_root=data_cfg.get("data_root", args.data_root),
        subset="testing",
        sample_rate=sample_rate,
        num_samples=int(data_cfg.get("num_samples", 16000)),
        download=False,
    )

    buffer, keyword_start = build_streaming_buffer(
        dataset, args.target, sample_rate=sample_rate, rng=rng
    )

    triggered, trigger_time, timeline = run_streaming_inference(
        model=model,
        feature_extractor=feature_extractor,
        waveform_buffer=buffer,
        sample_rate=sample_rate,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        smooth_window=args.smooth_window,
        trigger_consecutive=args.trigger_consecutive,
        target_index=target_index,
        threshold=args.threshold,
        device=device,
    )

    result = {
        "target": args.target,
        "threshold": args.threshold,
        "triggered": triggered,
        "trigger_time_sec": round(trigger_time, 2) if trigger_time is not None else None,
        "keyword_ground_truth_start_sec": round(keyword_start, 2),
        "timeline": timeline,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"target: {args.target}")
    print(f"threshold: {args.threshold}")
    print(f"triggered: {triggered}")
    print(f"trigger_time_sec: {result['trigger_time_sec']}")
    print(f"keyword_ground_truth_start_sec: {result['keyword_ground_truth_start_sec']}")
    print(f"saved stream demo: {output_path}")


if __name__ == "__main__":
    main()
