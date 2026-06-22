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
from src.models import build_model, count_parameters, normalize_model_config


WARMUP = 50
REPEAT = 500
INPUT_SHAPE = (1, 1, 40, 101)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark KWS inference latency on CPU.")
    parser.add_argument("--ckpt", required=True, help="PyTorch checkpoint path.")
    parser.add_argument("--onnx", required=True, help="ONNX FP32 model path.")
    parser.add_argument("--onnx-int8", required=True, help="ONNX INT8 model path.")
    parser.add_argument("--output", default="outputs/latency.json", help="Output JSON path.")

    parser.add_argument("--warmup", type=int, default=WARMUP, help="Warmup iterations.")
    parser.add_argument("--repeat", type=int, default=REPEAT, help="Benchmark iterations.")

    return parser.parse_args()


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def build_pytorch_model(checkpoint: dict[str, Any]) -> tuple[torch.nn.Module, torch.Tensor]:
    feature_cfg = dict(checkpoint.get("feature_config", {}))
    model_cfg = normalize_model_config(dict(checkpoint.get("model_config", {})), feature_cfg)
    model = build_model(model_cfg, feature_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dummy = torch.randn(INPUT_SHAPE, dtype=torch.float32)
    return model, dummy


def benchmark_pytorch(model: torch.nn.Module, dummy: torch.Tensor, warmup: int, repeat: int) -> list[float]:
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

        times: list[float] = []
        for _ in range(repeat):
            start = time.perf_counter()
            _ = model(dummy)
            end = time.perf_counter()
            times.append((end - start) * 1000.0)
    return times


def benchmark_onnx(onnx_path: Path, warmup: int, repeat: int) -> list[float]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)

    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy})
    times: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        _ = session.run(None, {input_name: dummy})
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
    return times



def summarize(times: list[float], model_path: Path | None = None) -> dict[str, Any]:
    arr = np.asarray(times, dtype=np.float64)
    result: dict[str, Any] = {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
    }
    if model_path is not None and model_path.exists():
        result["model_size_mb"] = model_path.stat().st_size / (1024 * 1024)
    else:
        result["model_size_mb"] = None
    return result


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.ckpt)
    pt_model, pt_dummy = build_pytorch_model(checkpoint)

    print(f"benchmark input shape: {INPUT_SHAPE}")
    print(f"warmup={args.warmup}, repeat={args.repeat}")

    pt_times = benchmark_pytorch(pt_model, pt_dummy, args.warmup, args.repeat)
    onnx_times = benchmark_onnx(Path(args.onnx), args.warmup, args.repeat)
    int8_times = benchmark_onnx(Path(args.onnx_int8), args.warmup, args.repeat)

    latency = {
        "pytorch_fp32": summarize(pt_times, Path(args.ckpt)),
        "onnx_fp32": summarize(onnx_times, Path(args.onnx)),
        "onnx_int8": summarize(int8_times, Path(args.onnx_int8)),
        "metadata": {
            "input_shape": list(INPUT_SHAPE),
            "warmup": args.warmup,
            "repeat": args.repeat,
            "num_parameters": count_parameters(pt_model),
            "device": "cpu",
        },
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(latency, file, indent=2)

    print(f"pytorch_fp32: mean={latency['pytorch_fp32']['mean_ms']:.4f} ms, "
          f"p50={latency['pytorch_fp32']['p50_ms']:.4f} ms, "
          f"p95={latency['pytorch_fp32']['p95_ms']:.4f} ms")
    print(f"onnx_fp32:    mean={latency['onnx_fp32']['mean_ms']:.4f} ms, "
          f"p50={latency['onnx_fp32']['p50_ms']:.4f} ms, "
          f"p95={latency['onnx_fp32']['p95_ms']:.4f} ms")
    print(f"onnx_int8:    mean={latency['onnx_int8']['mean_ms']:.4f} ms, "
          f"p50={latency['onnx_int8']['p50_ms']:.4f} ms, "
          f"p95={latency['onnx_int8']['p95_ms']:.4f} ms")
    print(f"saved latency: {output_path}")


if __name__ == "__main__":
    main()
