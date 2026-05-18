from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.speech_commands import ALL_LABELS, SpeechCommandsKWS, summarize_labels
from src.features.logmel import LogMelExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Speech Commands KWS data loading.")
    parser.add_argument(
        "--config",
        default="configs/mka_smallcnn.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--data-root", default=None, help="Override dataset root directory.")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum real examples per split.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override inspection batch size.")
    parser.add_argument("--no-download", action="store_true", help="Disable torchaudio dataset download.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def make_dataset(
    subset: str,
    data_cfg: dict[str, Any],
    limit: int,
    download: bool,
    seed: int,
) -> SpeechCommandsKWS:
    return SpeechCommandsKWS(
        data_root=data_cfg.get("data_root", "data/SpeechCommands"),
        subset=subset,
        sample_rate=int(data_cfg.get("sample_rate", 16_000)),
        num_samples=int(data_cfg.get("num_samples", 16_000)),
        limit=limit,
        download=download,
        silence_ratio=float(data_cfg.get("silence_ratio", 0.05)),
        seed=seed,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = dict(config.get("data", {}))
    feature_cfg = dict(config.get("features", {}))
    train_cfg = dict(config.get("train", {}))
    seed = int(config.get("seed", 1337))

    if args.data_root is not None:
        data_cfg["data_root"] = args.data_root

    batch_size = int(args.batch_size or train_cfg.get("batch_size", 32))
    splits = ("training", "validation", "testing")
    split_summaries: dict[str, Any] = {}
    datasets: dict[str, SpeechCommandsKWS] = {}

    for offset, split in enumerate(splits):
        dataset = make_dataset(
            subset=split,
            data_cfg=data_cfg,
            limit=args.limit,
            download=not args.no_download,
            seed=seed + offset,
        )
        if len(dataset) == 0:
            raise RuntimeError(f"{split} dataset is empty.")
        datasets[split] = dataset
        counts = summarize_labels(dataset)
        real_examples = len(dataset.indices)
        split_summaries[split] = {
            "real_examples": real_examples,
            "examples_including_silence": len(dataset),
            "class_distribution": counts,
        }

    train_loader = DataLoader(
        datasets["training"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    waveforms, targets = next(iter(train_loader))
    extractor = LogMelExtractor(**feature_cfg)
    with torch.no_grad():
        features = extractor(waveforms)

    label_mapping = {label: idx for idx, label in enumerate(ALL_LABELS)}
    summary: dict[str, Any] = {
        "config": str(Path(args.config)),
        "data_root": str(Path(data_cfg.get("data_root", "data/SpeechCommands"))),
        "limit": args.limit,
        "sample_rate": int(data_cfg.get("sample_rate", 16_000)),
        "fixed_waveform_length": int(data_cfg.get("num_samples", 16_000)),
        "label_mapping": label_mapping,
        "splits": split_summaries,
        "batch": {
            "batch_size": int(waveforms.shape[0]),
            "waveform_shape": list(waveforms.shape),
            "logmel_feature_shape": list(features.shape),
            "target_shape": list(targets.shape),
        },
    }

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / "data_summary.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"config: {summary['config']}")
    print(f"dataset_root: {summary['data_root']}")
    print(f"sample_rate: {summary['sample_rate']}")
    print(f"fixed_waveform_length: {summary['fixed_waveform_length']}")
    print("label_mapping:")
    for label, idx in label_mapping.items():
        print(f"  {idx}: {label}")
    print("splits:")
    for split in splits:
        split_summary = split_summaries[split]
        print(
            f"  {split}: real_examples={split_summary['real_examples']} "
            f"examples_including_silence={split_summary['examples_including_silence']}"
        )
        print("    class_distribution:")
        for label in ALL_LABELS:
            print(f"      {label}: {split_summary['class_distribution'][label]}")
    print(f"batch_waveform_shape: {tuple(waveforms.shape)}")
    print(f"batch_logmel_feature_shape: {tuple(features.shape)}")
    print(f"saved data summary: {output_path}")


if __name__ == "__main__":
    main()
