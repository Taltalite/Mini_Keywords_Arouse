from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.speech_commands import ALL_LABELS, TARGET_WORDS, SpeechCommandsKWS
from src.features.logmel import LogMelExtractor
from src.models import build_model, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SmallCNN KWS checkpoint.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path, e.g. outputs/best.pt.")
    parser.add_argument(
        "--subset",
        default="testing",
        choices=["validation", "testing"],
        help="Speech Commands subset to evaluate.",
    )
    parser.add_argument("--data-root", default=None, help="Override dataset root.")
    parser.add_argument("--limit", type=int, default=None, help="Limit real evaluation examples.")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional evaluation output directory. It must be absent or empty when provided.",
    )
    parser.add_argument("--no-download", action="store_true", help="Disable torchaudio dataset download.")
    return parser.parse_args()


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def write_confusion_csv(path: Path, labels: list[str], matrix: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true_label", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def mean_available(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    if not available:
        return None
    return sum(available) / len(available)


def build_silence_error_analysis(
    labels: list[str],
    matrix: list[list[int]],
    subset: str,
    limit: int | None,
) -> dict[str, Any]:
    silence_idx = labels.index("silence")
    silence_row = matrix[silence_idx]
    total = sum(silence_row)
    predicted_as = {
        label: {
            "count": int(count),
            "ratio": (count / total if total else None),
        }
        for label, count in zip(labels, silence_row)
    }
    top_predictions = sorted(
        (
            {
                "label": label,
                "count": int(count),
                "ratio": (count / total if total else None),
            }
            for label, count in zip(labels, silence_row)
        ),
        key=lambda item: item["count"],
        reverse=True,
    )
    unknown_count = silence_row[labels.index("unknown")]
    return {
        "subset": subset,
        "limit": limit,
        "silence_ground_truth_count": int(total),
        "silence_correct_count": int(silence_row[silence_idx]),
        "silence_accuracy": (silence_row[silence_idx] / total if total else None),
        "silence_confusion_row": {label: int(count) for label, count in zip(labels, silence_row)},
        "silence_predicted_as": predicted_as,
        "silence_top3_predictions": top_predictions[:3],
        "silence_predicted_unknown_count": int(unknown_count),
        "silence_predicted_unknown_ratio": (unknown_count / total if total else None),
        "silence_mainly_predicted_as_unknown": bool(total and unknown_count / total >= 0.5),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output) if args.output is not None else Path("outputs")
    if args.output is not None:
        if output_dir.exists() and not output_dir.is_dir():
            print(f"error: --output must be a directory path, got existing file: {output_dir}")
            sys.exit(-1)
        if output_dir.exists() and any(output_dir.iterdir()):
            print(
                f"warning: output directory is not empty and files may be overwritten: {output_dir}"
            )
            sys.exit(-1)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.ckpt)
    labels = list(checkpoint.get("labels", ALL_LABELS))
    data_cfg = dict(checkpoint.get("data_config", {}))
    feature_cfg = dict(checkpoint.get("feature_config", {}))
    model_cfg = dict(checkpoint.get("model_config", {}))

    if args.data_root is not None:
        data_cfg["data_root"] = args.data_root

    eval_dataset = SpeechCommandsKWS(
        data_root=data_cfg.get("data_root", "data/SpeechCommands"),
        subset=args.subset,
        sample_rate=int(data_cfg.get("sample_rate", 16_000)),
        num_samples=int(data_cfg.get("num_samples", 16_000)),
        limit=args.limit,
        download=not args.no_download,
        silence_ratio=float(data_cfg.get("silence_ratio", 0.05)),
        silence_gain_min=float(data_cfg.get("silence_gain_min", 0.0)),
        silence_gain_max=float(data_cfg.get("silence_gain_max", 0.001)),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    device = torch.device("cpu")
    feature_extractor = LogMelExtractor(**feature_cfg).to(device)
    model_cfg.setdefault("name", "small_cnn")
    model = build_model(model_cfg, feature_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    num_parameters = count_parameters(model)

    num_classes = len(labels)
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    prediction_counts = {label: 0 for label in labels}
    ground_truth_counts = {label: 0 for label in labels}
    correct = 0
    total = 0

    with torch.no_grad():
        for waveforms, targets in eval_loader:
            waveforms = waveforms.to(device)
            targets = targets.to(device)
            logits = model(feature_extractor(waveforms))
            preds = logits.argmax(dim=1)
            correct += int((preds == targets).sum().item())
            total += int(targets.numel())
            for true_idx, pred_idx in zip(targets.tolist(), preds.tolist()):
                matrix[int(true_idx)][int(pred_idx)] += 1
                ground_truth_counts[labels[int(true_idx)]] += 1
                prediction_counts[labels[int(pred_idx)]] += 1

    eval_accuracy = correct / total if total else 0.0
    per_class_accuracy: dict[str, float | None] = {}
    for idx, label in enumerate(labels):
        row_total = sum(matrix[idx])
        per_class_accuracy[label] = matrix[idx][idx] / row_total if row_total else None
    macro_accuracy = mean_available(list(per_class_accuracy.values()))
    target_keyword_macro_accuracy = mean_available(
        [per_class_accuracy.get(label) for label in TARGET_WORDS]
    )
    non_unknown_macro_accuracy = mean_available(
        [value for label, value in per_class_accuracy.items() if label != "unknown"]
    )

    confusion_path = output_dir / f"confusion_matrix_{args.subset}.csv"
    write_confusion_csv(confusion_path, labels, matrix)
    distribution_path = output_dir / f"prediction_distribution_{args.subset}.json"
    distribution_payload = {
        "subset": args.subset,
        "limit": args.limit,
        "prediction_distribution": prediction_counts,
        "ground_truth_distribution": ground_truth_counts,
    }
    with distribution_path.open("w", encoding="utf-8") as file:
        json.dump(distribution_payload, file, indent=2)
    silence_analysis = build_silence_error_analysis(labels, matrix, args.subset, args.limit)
    silence_analysis_path = output_dir / f"silence_error_analysis_{args.subset}.json"
    with silence_analysis_path.open("w", encoding="utf-8") as file:
        json.dump(silence_analysis, file, indent=2)
    legacy_silence_analysis_path = Path("outputs") / f"silence_error_analysis_{args.subset}.json"
    legacy_silence_analysis_path.parent.mkdir(parents=True, exist_ok=True)
    if legacy_silence_analysis_path.resolve() != silence_analysis_path.resolve():
        with legacy_silence_analysis_path.open("w", encoding="utf-8") as file:
            json.dump(silence_analysis, file, indent=2)

    metrics_path = output_dir / "metrics.json"
    metrics = {}
    metrics["output_dir"] = str(output_dir)
    metrics["macro_accuracy"] = macro_accuracy
    metrics["target_keyword_macro_accuracy"] = target_keyword_macro_accuracy
    metrics["non_unknown_macro_accuracy"] = non_unknown_macro_accuracy
    metrics["prediction_distribution"] = prediction_counts
    metrics["ground_truth_distribution"] = ground_truth_counts
    metrics["prediction_distribution_json"] = str(distribution_path)
    metrics["silence_error_analysis_json"] = str(silence_analysis_path)
    metrics["silence_error_analysis_legacy_json"] = str(legacy_silence_analysis_path)
    if args.subset == "testing":
        metrics["test_accuracy"] = eval_accuracy
        metrics["per_class_accuracy"] = per_class_accuracy
        metrics["confusion_matrix_csv"] = str(confusion_path)
        metrics["testing_accuracy"] = eval_accuracy
        metrics["testing_per_class_accuracy"] = per_class_accuracy
        metrics["testing_confusion_matrix_csv"] = str(confusion_path)
        metrics["testing_macro_accuracy"] = macro_accuracy
        metrics["testing_target_keyword_macro_accuracy"] = target_keyword_macro_accuracy
        metrics["testing_non_unknown_macro_accuracy"] = non_unknown_macro_accuracy
        metrics["testing_prediction_distribution"] = prediction_counts
        metrics["testing_ground_truth_distribution"] = ground_truth_counts
        metrics["testing_prediction_distribution_json"] = str(distribution_path)
        metrics["testing_silence_error_analysis_json"] = str(silence_analysis_path)
    else:
        metrics["validation_accuracy"] = eval_accuracy
        metrics["validation_per_class_accuracy"] = per_class_accuracy
        metrics["validation_confusion_matrix_csv"] = str(confusion_path)
        metrics["validation_macro_accuracy"] = macro_accuracy
        metrics["validation_target_keyword_macro_accuracy"] = target_keyword_macro_accuracy
        metrics["validation_non_unknown_macro_accuracy"] = non_unknown_macro_accuracy
        metrics["validation_prediction_distribution"] = prediction_counts
        metrics["validation_ground_truth_distribution"] = ground_truth_counts
        metrics["validation_prediction_distribution_json"] = str(distribution_path)
        metrics["validation_silence_error_analysis_json"] = str(silence_analysis_path)
    metrics["num_parameters"] = num_parameters
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"{args.subset}_accuracy={eval_accuracy:.4f}")
    print(f"macro_accuracy={'null' if macro_accuracy is None else f'{macro_accuracy:.4f}'}")
    print(
        "target_keyword_macro_accuracy="
        f"{'null' if target_keyword_macro_accuracy is None else f'{target_keyword_macro_accuracy:.4f}'}"
    )
    print(
        "non_unknown_macro_accuracy="
        f"{'null' if non_unknown_macro_accuracy is None else f'{non_unknown_macro_accuracy:.4f}'}"
    )
    print(f"num_parameters={num_parameters}")
    print(f"ground_truth_distribution: {ground_truth_counts}")
    print(f"prediction_distribution: {prediction_counts}")
    print(f"{args.subset}_silence_ground_truth_count={silence_analysis['silence_ground_truth_count']}")
    print(f"{args.subset}_silence_top3_predictions={silence_analysis['silence_top3_predictions']}")
    print(
        f"{args.subset}_silence_mainly_predicted_as_unknown="
        f"{silence_analysis['silence_mainly_predicted_as_unknown']}"
    )
    print("per_class_accuracy:")
    for label in labels:
        value = per_class_accuracy[label]
        text = "null" if value is None else f"{value:.4f}"
        print(f"  {label}: {text}")
    print(f"saved confusion matrix: {confusion_path}")
    print(f"saved prediction distribution: {distribution_path}")
    print(f"saved silence error analysis: {silence_analysis_path}")
    print(f"updated metrics: {metrics_path}")


if __name__ == "__main__":
    main()
