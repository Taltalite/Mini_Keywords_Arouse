from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate model card and resume bullets from metrics.")
    parser.add_argument("--metrics", default="outputs/metrics.json", help="Path to metrics JSON.")
    parser.add_argument("--latency", default="outputs/latency.json", help="Path to latency JSON.")
    parser.add_argument("--stream", default="outputs/stream_demo.json", help="Path to stream demo JSON.")
    parser.add_argument("--output-dir", default="reports", help="Output directory for generated reports.")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt_percent(value: float | None) -> str:
    if value is None:
        return "TODO"
    return f"{value * 100:.2f}%"


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "TODO"
    return f"{value:.2f}"


def fmt_mb(value: float | None) -> str:
    if value is None:
        return "TODO"
    return f"{value:.2f}"


def generate_model_card(metrics: dict, latency: dict, stream: dict) -> str:
    test_acc = metrics.get("test_accuracy") or metrics.get("testing_accuracy")
    val_acc = metrics.get("best_val_accuracy") or metrics.get("validation_accuracy")
    num_params = metrics.get("num_parameters")
    model_name = metrics.get("model_name", "TODO")
    epochs = metrics.get("epochs", "TODO")

    per_class = metrics.get("testing_per_class_accuracy") or metrics.get("per_class_accuracy") or {}
    silence_acc = per_class.get("silence")

    lines = [
        "# Model Card",
        "",
        "## Overview",
        f"- **Model**: {model_name}",
        f"- **Parameters**: {num_params:,}" if num_params else "- **Parameters**: TODO",
        f"- **Training Epochs**: {epochs}",
        f"- **Best Validation Accuracy**: {fmt_percent(val_acc)}",
        f"- **Testing Accuracy**: {fmt_percent(test_acc)}",
        f"- **Silence Accuracy**: {fmt_percent(silence_acc)}",
        "",
        "## Per-Class Testing Accuracy",
        "",
    ]

    for label, acc in sorted((per_class or {}).items()):
        lines.append(f"- {label}: {fmt_percent(acc)}")

    lines.extend([
        "",
        "## Latency Benchmark (CPU, batch=1)",
        "",
        "| Backend | Mean (ms) | p50 (ms) | p95 (ms) | Size (MB) |",
        "|---|---|---|---|---|",
        f"| PyTorch FP32 | {fmt_ms(latency.get('pytorch_fp32', {}).get('mean_ms'))} | {fmt_ms(latency.get('pytorch_fp32', {}).get('p50_ms'))} | {fmt_ms(latency.get('pytorch_fp32', {}).get('p95_ms'))} | {fmt_mb(latency.get('pytorch_fp32', {}).get('model_size_mb'))} |",
        f"| ONNX FP32 | {fmt_ms(latency.get('onnx_fp32', {}).get('mean_ms'))} | {fmt_ms(latency.get('onnx_fp32', {}).get('p50_ms'))} | {fmt_ms(latency.get('onnx_fp32', {}).get('p95_ms'))} | {fmt_mb(latency.get('onnx_fp32', {}).get('model_size_mb'))} |",
        f"| ONNX INT8 | {fmt_ms(latency.get('onnx_int8', {}).get('mean_ms'))} | {fmt_ms(latency.get('onnx_int8', {}).get('p50_ms'))} | {fmt_ms(latency.get('onnx_int8', {}).get('p95_ms'))} | {fmt_mb(latency.get('onnx_int8', {}).get('model_size_mb'))} |",
        "",
        "## Streaming Inference",
        "",
        f"- **Target**: {stream.get('target', 'TODO')}",
        f"- **Threshold**: {stream.get('threshold', 'TODO')}",
        f"- **Triggered**: {stream.get('triggered', 'TODO')}",
        f"- **Trigger Time (sec)**: {stream.get('trigger_time_sec') if stream.get('trigger_time_sec') is not None else 'TODO'}",
        "",
        "## Notes",
        "- All metrics are measured on WSL2 CPU unless otherwise stated.",
        "- INT8 quantization reduced model size but increased latency due to small model size (177K params) and CPU QDQ overhead.",
    ])

    return "\n".join(lines)


def generate_resume_bullets(metrics: dict, latency: dict, stream: dict) -> str:
    test_acc = metrics.get("test_accuracy") or metrics.get("testing_accuracy")
    val_acc = metrics.get("best_val_accuracy") or metrics.get("validation_accuracy")
    num_params = metrics.get("num_parameters")
    model_name = metrics.get("model_name", "TODO")

    per_class = metrics.get("testing_per_class_accuracy") or metrics.get("per_class_accuracy") or {}
    silence_acc = per_class.get("silence")

    onnx_fp32 = latency.get("onnx_fp32", {})
    onnx_int8 = latency.get("onnx_int8", {})

    params_str = f"{num_params:,}" if num_params is not None else "TODO"
    lines = [
        "# Resume Bullets",
        "",
        "## Keyword Spotting System",
        "",
        f"- Built a lightweight keyword spotting system on Google Speech Commands V2, including waveform loading, 16 kHz resampling, log-Mel feature extraction, {model_name} training, and evaluation pipeline; achieved {fmt_percent(test_acc)} test accuracy with {params_str} parameters.",
        f"- Implemented PyTorch → ONNX → ONNX Runtime deployment and INT8 dynamic quantization; reduced model size from {fmt_mb(onnx_fp32.get('model_size_mb'))} MB to {fmt_mb(onnx_int8.get('model_size_mb'))} MB and measured CPU p95 latency of {fmt_ms(onnx_fp32.get('p95_ms'))} ms at batch size 1.",
        f"- Designed a sliding-window streaming inference strategy with confidence smoothing and threshold-based triggering, simulating low-latency keyword detection on continuous audio streams; silence classification accuracy reached {fmt_percent(silence_acc)}.",
        "- Developed a real-time microphone demo prototype (`src/mic_demo.py`) for Windows, demonstrating end-to-end feasibility from audio capture to model inference.",
        "",
        "## Quantitative Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Testing Accuracy | {fmt_percent(test_acc)} |",
        f"| Best Validation Accuracy | {fmt_percent(val_acc)} |",
        f"| Parameters | {params_str} |",
        f"| ONNX FP32 p95 Latency | {fmt_ms(onnx_fp32.get('p95_ms'))} ms |",
        f"| ONNX INT8 Model Size | {fmt_mb(onnx_int8.get('model_size_mb'))} MB |",
        f"| Streaming Trigger Latency | {stream.get('trigger_time_sec') if stream.get('trigger_time_sec') is not None else 'TODO'} sec |",
    ]

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    metrics = load_json(args.metrics)
    latency = load_json(args.latency)
    stream = load_json(args.stream)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_card = generate_model_card(metrics, latency, stream)
    model_card_path = output_dir / "model_card.md"
    with model_card_path.open("w", encoding="utf-8") as f:
        f.write(model_card)

    resume_bullets = generate_resume_bullets(metrics, latency, stream)
    resume_path = output_dir / "resume_bullets.md"
    with resume_path.open("w", encoding="utf-8") as f:
        f.write(resume_bullets)

    print(f"generated model card: {model_card_path}")
    print(f"generated resume bullets: {resume_path}")


if __name__ == "__main__":
    main()
