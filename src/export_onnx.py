from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.features.logmel import LogMelExtractor
from src.models import build_model, count_parameters, normalize_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PyTorch KWS checkpoint to ONNX.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path, e.g. outputs/best.pt.")
    parser.add_argument("--config", default=None, help="Optional YAML config override.")
    parser.add_argument("--out", required=True, help="Output ONNX path, e.g. outputs/model.onnx.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    return parser.parse_args()


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def infer_feature_shape(feature_cfg: dict[str, Any], data_cfg: dict[str, Any]) -> torch.Tensor:
    num_samples = int(data_cfg.get("num_samples", 16_000))
    extractor = LogMelExtractor(**feature_cfg).eval()
    dummy_waveform = torch.zeros(1, 1, num_samples, dtype=torch.float32)
    with torch.no_grad():
        return extractor(dummy_waveform)


def export_onnx(
    model: torch.nn.Module,
    dummy_features: torch.Tensor,
    output_path: Path,
    opset: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model,
        dummy_features,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["logmel"],
        output_names=["logits"],
        dynamic_axes=None,
        dynamo=False,
    )


def run_onnx_validation(
    model: torch.nn.Module,
    dummy_features: torch.Tensor,
    onnx_path: Path,
) -> dict[str, Any]:
    with torch.no_grad():
        torch_logits = model(dummy_features).detach().cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_logits = session.run(None, {input_name: dummy_features.detach().cpu().numpy()})[0]

    diff = np.abs(torch_logits - onnx_logits)
    return {
        "onnx_input_name": input_name,
        "torch_output_shape": list(torch_logits.shape),
        "onnx_output_shape": list(onnx_logits.shape),
        "max_abs_diff": float(diff.max()) if diff.size else None,
        "mean_abs_diff": float(diff.mean()) if diff.size else None,
    }


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.ckpt)

    feature_cfg = dict(checkpoint.get("feature_config", {}))
    data_cfg = dict(checkpoint.get("data_config", {}))
    model_cfg = dict(checkpoint.get("model_config", {}))

    if args.config is not None:
        import yaml

        with Path(args.config).open("r", encoding="utf-8") as f:
            external = yaml.safe_load(f) or {}
        feature_cfg.update(dict(external.get("features", {})))
        data_cfg.update(dict(external.get("data", {})))
        model_cfg.update(dict(external.get("model", {})))

    model_cfg.setdefault("name", "small_cnn")
    model_cfg = normalize_model_config(model_cfg, feature_cfg)

    model = build_model(model_cfg, feature_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_features = infer_feature_shape(feature_cfg, data_cfg)
    output_path = Path(args.out)
    export_onnx(model, dummy_features, output_path, args.opset)

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    validation = run_onnx_validation(model, dummy_features, output_path)

    check = {
        "ckpt": str(Path(args.ckpt)),
        "onnx_path": str(output_path),
        "opset": args.opset,
        "feature_input_shape": list(dummy_features.shape),
        "num_parameters": count_parameters(model),
        "onnx_file_size_bytes": output_path.stat().st_size,
        "onnxruntime_provider": "CPUExecutionProvider",
        "status": "passed",
        **validation,
    }

    check_path = output_path.parent / "onnx_check.json"
    with check_path.open("w", encoding="utf-8") as f:
        json.dump(check, f, indent=2)

    print(f"exported onnx: {output_path}")
    print(f"feature_input_shape: {tuple(dummy_features.shape)}")
    print(f"onnx_output_shape: {tuple(check['onnx_output_shape'])}")
    print(f"max_abs_diff={check['max_abs_diff']:.8f}")
    print(f"mean_abs_diff={check['mean_abs_diff']:.8f}")
    print(f"saved onnx check: {check_path}")


if __name__ == "__main__":
    main()
