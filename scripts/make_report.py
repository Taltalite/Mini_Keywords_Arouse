from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
DOCS_DIR = ROOT / "docs"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return {}


def get(metrics: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = metrics
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def fmt(value: Any, precision: int = 4, percent: bool = False) -> str:
    if value is None:
        return "TODO"
    if isinstance(value, float):
        if percent:
            return f"{value * 100:.{precision}f}%"
        return f"{value:.{precision}f}"
    return str(value)


def render_model_card(metrics: dict[str, Any], latency: dict[str, Any], stream: dict[str, Any]) -> str:
    best_val = get(metrics, "best_val_accuracy")
    test_acc = get(metrics, "test_accuracy")
    num_params = get(metrics, "num_parameters")
    model_name = get(metrics, "model_name", default="SmallCNN")
    epochs = get(metrics, "epochs")
    limit = get(metrics, "limit")

    pytorch = latency.get("pytorch_fp32", {})
    onnx_fp32 = latency.get("onnx_fp32", {})
    onnx_int8 = latency.get("onnx_int8", {})

    fp32_size = get(onnx_fp32, "model_size_mb")
    int8_size = get(onnx_int8, "model_size_mb")

    lines = [
        "# Model Card: Mini Keywords Arouse",
        "",
        "## Task",
        "",
        "12-class keyword spotting on Google Speech Commands V2:",
        "`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`, `unknown`, `silence`.",
        "",
        "## Model",
        "",
        f"- Architecture: `{model_name}`",
        f"- Parameters: {fmt(num_params, 0)}",
        f"- Input shape: `[1, 1, 40, 101]` (log-Mel spectrogram)",
        f"- Training epochs: {fmt(epochs, 0)}",
        f"- Training subset limit: {fmt(limit, 0) if limit is not None else 'full'}",
        "",
        "## Metrics",
        "",
        f"- Best validation accuracy: {fmt(best_val)}",
        f"- Test accuracy: {fmt(test_acc)}",
        "",
        "## Latency Benchmark (CPU, batch size 1)",
        "",
        "| Backend | Mean (ms) | p50 (ms) | p95 (ms) | Model size (MB) |",
        "|---|---|---|---|---|",
        f"| PyTorch FP32 | {fmt(get(pytorch, 'mean_ms'))} | {fmt(get(pytorch, 'p50_ms'))} | {fmt(get(pytorch, 'p95_ms'))} | {fmt(get(pytorch, 'model_size_mb'))} |",
        f"| ONNX FP32 | {fmt(get(onnx_fp32, 'mean_ms'))} | {fmt(get(onnx_fp32, 'p50_ms'))} | {fmt(get(onnx_fp32, 'p95_ms'))} | {fmt(get(onnx_fp32, 'model_size_mb'))} |",
        f"| ONNX INT8 | {fmt(get(onnx_int8, 'mean_ms'))} | {fmt(get(onnx_int8, 'p50_ms'))} | {fmt(get(onnx_int8, 'p95_ms'))} | {fmt(get(onnx_int8, 'model_size_mb'))} |",
        "",
        "## Quantization",
        "",
        f"- FP32 model size: {fmt(fp32_size)} MB",
        f"- INT8 model size: {fmt(int8_size)} MB",
        "- Method: ONNX Runtime dynamic quantization (`weight_type=QInt8`)",
        "",
        "## Streaming Inference Demo",
        "",
        f"- Target keyword: `{fmt(get(stream, 'target'))}`",
        f"- Threshold: {fmt(get(stream, 'threshold'))}",
        f"- Triggered: {fmt(get(stream, 'triggered'))}",
        f"- Trigger time (sec): {fmt(get(stream, 'trigger_time_sec'))}",
        f"- Keyword insertion time (sec): {fmt(get(stream, 'keyword_insertion_time_sec'))}",
        f"- Timeline windows: {fmt(get(stream, 'timeline', default=[]).__len__(), 0)}",
        "",
        "## Notes",
        "",
        "All numbers are measured on the local WSL2 CPU environment. "
        "Missing values are reported as `TODO` rather than fabricated.",
        "",
    ]
    return "\n".join(lines)


def render_resume_bullets(metrics: dict[str, Any], latency: dict[str, Any], stream: dict[str, Any]) -> str:
    best_val = get(metrics, "best_val_accuracy")
    test_acc = get(metrics, "test_accuracy")
    num_params = get(metrics, "num_parameters")

    onnx_fp32 = latency.get("onnx_fp32", {})
    onnx_int8 = latency.get("onnx_int8", {})
    fp32_size = get(onnx_fp32, "model_size_mb")
    int8_size = get(onnx_int8, "model_size_mb")
    int8_p95 = get(onnx_int8, "p95_ms")

    lines = [
        "# Resume Bullets",
        "",
        "- Built a lightweight keyword spotting system on Google Speech Commands V2, including waveform loading, "
        "16 kHz resampling, log-Mel feature extraction, SmallCNN training, and evaluation pipeline; "
        f"achieved {fmt(test_acc, precision=2, percent=True)} test accuracy with {fmt(num_params, 0)} parameters.",
        "",
        "- Implemented PyTorch → ONNX → ONNX Runtime deployment and INT8 dynamic quantization; "
        f"reduced model size from {fmt(fp32_size)} MB to {fmt(int8_size)} MB and measured CPU p95 latency of "
        f"{fmt(int8_p95)} ms at batch size 1.",
        "",
        "- Designed a sliding-window streaming inference strategy with confidence smoothing and threshold-based triggering, "
        "simulating low-latency keyword detection on continuous audio streams.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    metrics = load_json(OUTPUTS_DIR / "metrics.json")
    latency = load_json(OUTPUTS_DIR / "latency.json")
    stream = load_json(OUTPUTS_DIR / "stream_demo.json")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    model_card = render_model_card(metrics, latency, stream)
    resume_bullets = render_resume_bullets(metrics, latency, stream)

    model_card_path = OUTPUTS_DIR / "model_card.md"
    resume_bullets_path = DOCS_DIR / "resume_bullets.md"

    with model_card_path.open("w", encoding="utf-8") as file:
        file.write(model_card)
    with resume_bullets_path.open("w", encoding="utf-8") as file:
        file.write(resume_bullets)

    print(f"saved model card: {model_card_path}")
    print(f"saved resume bullets: {resume_bullets_path}")


if __name__ == "__main__":
    main()
