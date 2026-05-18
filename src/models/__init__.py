from __future__ import annotations

from typing import Any

from torch import nn

from .small_cnn import SmallCNN, count_parameters
from .tc_resnet import TCResNet


def normalize_model_config(model_cfg: dict[str, Any], feature_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    feature_cfg = feature_cfg or {}
    normalized = dict(model_cfg)
    name = str(normalized.get("name", "small_cnn")).lower()
    normalized["name"] = name
    normalized["num_classes"] = int(normalized.get("num_classes", 12))
    normalized["dropout"] = float(normalized.get("dropout", 0.1))
    if name == "tc_resnet":
        normalized["n_mels"] = int(normalized.get("n_mels", feature_cfg.get("n_mels", 40)))
        normalized["channels"] = int(normalized.get("channels", 48))
        normalized["num_blocks"] = int(normalized.get("num_blocks", 4))
        normalized["kernel_size"] = int(normalized.get("kernel_size", 9))
    return normalized


def build_model(model_cfg: dict[str, Any], feature_cfg: dict[str, Any] | None = None) -> nn.Module:
    cfg = normalize_model_config(model_cfg, feature_cfg)
    name = cfg["name"]
    if name == "small_cnn":
        return SmallCNN(num_classes=cfg["num_classes"], dropout=cfg["dropout"])
    if name == "tc_resnet":
        return TCResNet(
            num_classes=cfg["num_classes"],
            n_mels=cfg["n_mels"],
            channels=cfg["channels"],
            num_blocks=cfg["num_blocks"],
            kernel_size=cfg["kernel_size"],
            dropout=cfg["dropout"],
        )
    raise ValueError(f"Unsupported model name: {name}")


__all__ = [
    "SmallCNN",
    "TCResNet",
    "build_model",
    "count_parameters",
    "normalize_model_config",
]
