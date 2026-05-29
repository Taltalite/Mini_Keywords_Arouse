from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from src.features.logmel import LogMelExtractor
from src.models import build_model, normalize_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CPU inference latency for KWS models.")
    parser.add_argument("--ckpt", required=True, help="PyTorch checkpoint path.")
    parser.add_argument("--onnx", required=True, help="ONNX FP32 model path.")
    parser.add_argument("--onnx-int8", required=True, help="ONNX INT8 model path.")
    parser.add_argument("--output", default="outputs/latency.json", help="Output JSON path.")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations.")
    parser.add_argument("--repeat", type=int, default=500, help="Benchmark iterations.")
    return parser.parse_args()


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def benchmark_pytorch(model: torch.nn.Module, dummy_input: torch.Tensor, warmup: int, repeat: int) -> list[float]:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    times: list[float] = []
    with torch.no_grad():
        for _ in range(repeat):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            times.append((end - start) * 1000.0)
    return times


def benchmark_onnx(session: ort.InferenceSession, dummy_input: np.ndarray, warmup: int, repeat: int) -> list[float]:
    input_name = session.get_inputs()[0].name
    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy_input})

    times: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
    return times


def compute_metrics(times: list[float]) -> dict[str, float]:
    arr = np.array(times)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def main() -> None:
    args = parse_args()

    checkpoint = load_checkpoint(args.ckpt)
    model_cfg = dict(checkpoint.get("model_config", {}))
    feature_cfg = dict(checkpoint.get("feature_config", {}))
    data_cfg = dict(checkpoint.get("data_config", {}))
    model_cfg.setdefault("name", "small_cnn")
    model_cfg = normalize_model_config(model_cfg, feature_cfg)

    model = build_model(model_cfg, feature_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Derive exact feature shape from config (same path as export_onnx)
    num_samples = int(data_cfg.get("num_samples", 16_000))
    extractor = LogMelExtractor(**feature_cfg).eval()
    with torch.no_grad():
        dummy_torch = extractor(torch.zeros(1, 1, num_samples, dtype=torch.float32))
    dummy_np = dummy_torch.numpy()

    pytorch_times = benchmark_pytorch(model, dummy_torch, args.warmup, args.repeat)
    pytorch_metrics = compute_metrics(pytorch_times)
    pytorch_metrics["model_size_mb"] = round(Path(args.ckpt).stat().st_size / (1024 * 1024), 2)

    session_fp32 = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    onnx_times = benchmark_onnx(session_fp32, dummy_np, args.warmup, args.repeat)
    onnx_metrics = compute_metrics(onnx_times)
    onnx_metrics["model_size_mb"] = round(Path(args.onnx).stat().st_size / (1024 * 1024), 2)

    session_int8 = ort.InferenceSession(args.onnx_int8, providers=["CPUExecutionProvider"])
    int8_times = benchmark_onnx(session_int8, dummy_np, args.warmup, args.repeat)
    int8_metrics = compute_metrics(int8_times)
    int8_metrics["model_size_mb"] = round(Path(args.onnx_int8).stat().st_size / (1024 * 1024), 2)

    results = {
        "pytorch_fp32": pytorch_metrics,
        "onnx_fp32": onnx_metrics,
        "onnx_int8": int8_metrics,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"saved latency results: {output_path}")
    for key, metrics in results.items():
        print(
            f"{key}: "
            f"mean={metrics['mean_ms']:.3f}ms "
            f"p50={metrics['p50_ms']:.3f}ms "
            f"p95={metrics['p95_ms']:.3f}ms "
            f"size={metrics['model_size_mb']:.2f}MB"
        )


if __name__ == "__main__":
    main()
