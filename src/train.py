from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.speech_commands import ALL_LABELS, SpeechCommandsKWS
from src.features.logmel import LogMelExtractor
from src.models import build_model, count_parameters, normalize_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SmallCNN keyword spotting model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--data-root", default=None, help="Override dataset root.")
    parser.add_argument("--limit", type=int, default=None, help="Limit real train/validation examples.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers.")
    parser.add_argument("--no-download", action="store_true", help="Disable torchaudio dataset download.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def evaluate(
    model: nn.Module,
    features: LogMelExtractor,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for waveforms, targets in loader:
            waveforms = waveforms.to(device)
            targets = targets.to(device)
            logits = model(features(waveforms))
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += int(targets.numel())
    return correct / total if total else 0.0


def build_weighted_sampler(dataset: SpeechCommandsKWS) -> WeightedRandomSampler:
    targets = dataset.target_indices()
    counts = {idx: 0 for idx in range(len(ALL_LABELS))}
    for target in targets:
        counts[int(target)] += 1
    weights = [1.0 / counts[int(target)] if counts[int(target)] else 0.0 for target in targets]
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def build_class_weights(label_counts: dict[str, int], device: torch.device) -> torch.Tensor:
    total = sum(label_counts.values())
    num_classes = len(ALL_LABELS)
    weights = []
    for label in ALL_LABELS:
        count = label_counts[label]
        weights.append(total / (num_classes * count) if count else 0.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 1337))
    set_seed(seed)

    data_cfg = dict(config.get("data", {}))
    feature_cfg = dict(config.get("features", {}))
    model_cfg = dict(config.get("model", {}))
    train_cfg = dict(config.get("train", {}))

    if args.data_root is not None:
        data_cfg["data_root"] = args.data_root
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 1))

    device = torch.device(str(train_cfg.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use device: cpu.")

    train_dataset = SpeechCommandsKWS(
        data_root=data_cfg.get("data_root", "data/SpeechCommands"),
        subset="training",
        sample_rate=int(data_cfg.get("sample_rate", 16_000)),
        num_samples=int(data_cfg.get("num_samples", 16_000)),
        limit=args.limit,
        download=not args.no_download,
        silence_ratio=float(data_cfg.get("silence_ratio", 0.05)),
        seed=seed,
    )
    val_dataset = SpeechCommandsKWS(
        data_root=data_cfg.get("data_root", "data/SpeechCommands"),
        subset="validation",
        sample_rate=int(data_cfg.get("sample_rate", 16_000)),
        num_samples=int(data_cfg.get("num_samples", 16_000)),
        limit=args.limit,
        download=not args.no_download,
        silence_ratio=float(data_cfg.get("silence_ratio", 0.05)),
        seed=seed + 1,
    )

    train_label_counts = train_dataset.label_counts()
    use_weighted_sampler = bool(train_cfg.get("use_weighted_sampler", False))
    use_class_weighted_loss = bool(train_cfg.get("use_class_weighted_loss", False))
    sampler = build_weighted_sampler(train_dataset) if use_weighted_sampler else None
    sampler_type = "weighted_random" if sampler is not None else "shuffle"

    print("train_class_counts:")
    for label in ALL_LABELS:
        print(f"  {label}: {train_label_counts[label]}")
    print(f"sampler: {sampler_type}")
    print(f"use_class_weighted_loss: {use_class_weighted_loss}")

    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    feature_extractor = LogMelExtractor(**feature_cfg).to(device)
    model_cfg = normalize_model_config(model_cfg, feature_cfg)
    model = build_model(model_cfg, feature_cfg).to(device)

    class_weights = build_class_weights(train_label_counts, device) if use_class_weighted_loss else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {
        "train_loss": [],
        "val_accuracy": [],
        "best_val_accuracy": None,
        "num_parameters": count_parameters(model),
        "model_name": model_cfg["name"],
        "epochs": epochs,
        "limit": args.limit,
        "sampler": {
            "type": sampler_type,
            "use_weighted_sampler": use_weighted_sampler,
            "use_class_weighted_loss": use_class_weighted_loss,
            "train_class_counts": train_label_counts,
        },
    }

    best_val = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0
        running_acc = 0.0

        for waveforms, targets in train_loader:
            waveforms = waveforms.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(feature_extractor(waveforms))
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_count = int(targets.numel())
            running_loss += float(loss.item()) * batch_count
            running_acc += accuracy(logits.detach(), targets) * batch_count
            running_count += batch_count

        train_loss = running_loss / running_count if running_count else 0.0
        train_acc = running_acc / running_count if running_count else 0.0
        val_acc = evaluate(model, feature_extractor, val_loader, device)
        metrics["train_loss"].append(train_loss)
        metrics["val_accuracy"].append(val_acc)

        print(
            f"epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            metrics["best_val_accuracy"] = val_acc
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": model_cfg,
                "feature_config": feature_cfg,
                "data_config": data_cfg,
                "labels": list(ALL_LABELS),
                "metrics": metrics,
                "config": config,
            }
            torch.save(checkpoint, outputs_dir / "best.pt")

    with (outputs_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(f"saved checkpoint: {outputs_dir / 'best.pt'}")
    print(f"saved metrics: {outputs_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
