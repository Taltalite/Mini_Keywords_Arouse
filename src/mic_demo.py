from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.features.logmel import LogMelExtractor
from src.models import build_model, normalize_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time keyword spotting demo from microphone.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path.")
    parser.add_argument("--target", default="yes", help="Target keyword to detect.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Detection threshold.")
    parser.add_argument("--device", type=int, default=None, help="Input audio device index.")
    parser.add_argument("--window-sec", type=float, default=1.0, help="Sliding window size in seconds.")
    parser.add_argument("--hop-sec", type=float, default=0.1, help="Hop size in seconds.")
    parser.add_argument("--smooth-window", type=int, default=5, help="Moving average window for confidence smoothing.")
    parser.add_argument("--trigger-consecutive", type=int, default=3, help="Required consecutive windows above threshold to trigger.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate.")
    return parser.parse_args()


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class RingBuffer:
    """Simple float32 ring buffer for mono audio."""

    def __init__(self, size: int) -> None:
        self.buffer = np.zeros(size, dtype=np.float32)
        self.size = size

    def extend(self, data: np.ndarray) -> None:
        n = len(data)
        if n >= self.size:
            self.buffer[:] = data[-self.size :]
            return
        self.buffer[:-n] = self.buffer[n:]
        self.buffer[-n:] = data


def main() -> None:
    args = parse_args()

    try:
        import sounddevice as sd
    except ImportError as exc:
        print("ERROR: sounddevice is required for microphone input.")
        print("Install it with: uv pip install sounddevice")
        raise SystemExit(1) from exc

    checkpoint = load_checkpoint(args.ckpt)
    model_cfg = dict(checkpoint.get("model_config", {}))
    feature_cfg = dict(checkpoint.get("feature_config", {}))
    model_cfg.setdefault("name", "small_cnn")
    model_cfg = normalize_model_config(model_cfg, feature_cfg)

    device = torch.device("cpu")
    feature_extractor = LogMelExtractor(**feature_cfg).to(device)
    model = build_model(model_cfg, feature_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels = list(checkpoint.get("labels", []))
    if args.target not in labels:
        print(f"ERROR: target '{args.target}' not in model labels: {labels}")
        raise SystemExit(1)
    target_index = labels.index(args.target)

    sample_rate = args.sample_rate
    num_window_samples = int(args.window_sec * sample_rate)
    num_hop_samples = int(args.hop_sec * sample_rate)

    ring_buffer = RingBuffer(num_window_samples)
    raw_probs: list[float] = []
    consecutive = 0

    def audio_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        nonlocal consecutive
        if status:
            print(f"Audio status: {status}", file=sys.stderr)

        # indata shape: [frames, channels]; take first channel as mono
        samples = indata[:, 0].astype(np.float32)
        ring_buffer.extend(samples)

        # Run inference on the current 1-second buffer
        waveform = torch.from_numpy(ring_buffer.buffer).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            features = feature_extractor(waveform)
            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            target_prob = probs[0, target_index].item()

        raw_probs.append(target_prob)
        if len(raw_probs) > args.smooth_window:
            raw_probs.pop(0)

        smooth_prob = sum(raw_probs) / len(raw_probs)

        if smooth_prob >= args.threshold:
            consecutive += 1
            if consecutive >= args.trigger_consecutive:
                print(f"Detected: {args.target}")
                consecutive = 0  # reset to avoid spam
        else:
            consecutive = 0

    print(f"Listening for keyword: '{args.target}'")
    print(f"Threshold: {args.threshold} | Press Ctrl+C to stop")

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            blocksize=num_hop_samples,
            device=args.device,
            channels=1,
            dtype=np.float32,
            callback=audio_callback,
        ):
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
