from __future__ import annotations

import argparse

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from onnxruntime.quantization import QuantType, quantize_dynamic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize an ONNX model to INT8 using dynamic quantization.")
    parser.add_argument("--input", required=True, help="Input FP32 ONNX model path.")
    parser.add_argument("--output", required=True, help="Output INT8 ONNX model path.")
    parser.add_argument(
        "--weight-type",
        default="QInt8",
        choices=["QInt8", "QUInt8"],
        help="Quantized weight type (default: QInt8).",
    )
    return parser.parse_args()


def quantize(input_path: Path, output_path: Path, weight_type: QuantType) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=weight_type,
    )


def validate_onnx_model(onnx_path: Path, input_shape: tuple[int, ...]) -> dict[str, Any]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_dtype = np.float32  # ONNX Runtime 量化后输入仍为 FP32

    dummy_input = np.random.randn(*input_shape).astype(input_dtype)
    outputs = session.run(None, {input_name: dummy_input})
    output_array = outputs[0]

    return {
        "input_name": input_name,
        "input_shape": list(input_shape),
        "output_name": session.get_outputs()[0].name,
        "output_shape": list(output_array.shape),
        "output_dtype": str(output_array.dtype),
        "providers": session.get_providers(),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input ONNX model not found: {input_path}")

    weight_type = QuantType.QInt8 if args.weight_type == "QInt8" else QuantType.QUInt8
    quantize(input_path, output_path, weight_type)

    # Validate with the expected log-Mel input shape used by this project.
    input_shape = (1, 1, 40, 101)
    validation = validate_onnx_model(output_path, input_shape)

    check = {
        "input_onnx": str(input_path),
        "output_onnx": str(output_path),
        "weight_type": args.weight_type,
        "status": "passed",
        "int8_file_size_bytes": output_path.stat().st_size,
        "fp32_file_size_bytes": input_path.stat().st_size,
        **validation,
    }

    check_path = output_path.parent / "quantization_check.json"
    with check_path.open("w", encoding="utf-8") as file:
        json.dump(check, file, indent=2)

    print(f"quantized onnx: {output_path}")
    print(f"weight_type: {args.weight_type}")
    print(f"fp32_size_bytes: {check['fp32_file_size_bytes']}")
    print(f"int8_size_bytes: {check['int8_file_size_bytes']}")
    print(f"output_shape: {tuple(check['output_shape'])}")
    print(f"saved quantization check: {check_path}")


if __name__ == "__main__":
    main()
