from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize ONNX model to INT8 using ONNX Runtime dynamic quantization."
    )
    parser.add_argument("--input", required=True, help="Input ONNX model path.")
    parser.add_argument("--output", required=True, help="Output INT8 ONNX model path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input ONNX model not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["MatMul", "Gemm", "Conv"],
    )

    input_size = input_path.stat().st_size
    output_size = output_path.stat().st_size
    reduction = input_size / output_size if output_size else 0.0

    print(f"quantized: {output_path}")
    print(f"original size: {input_size / 1024 / 1024:.2f} MB")
    print(f"quantized size: {output_size / 1024 / 1024:.2f} MB")
    print(f"reduction: {reduction:.2f}x")


if __name__ == "__main__":
    main()
