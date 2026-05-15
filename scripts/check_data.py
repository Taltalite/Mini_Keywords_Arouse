from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.speech_commands import ALL_LABELS, SpeechCommandsKWS, summarize_labels
from src.features.logmel import LogMelExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Speech Commands KWS data loading.")
    parser.add_argument("--data-root", default="data/SpeechCommands", help="Dataset root directory.")
    parser.add_argument(
        "--subset",
        default="training",
        choices=["training", "validation", "testing"],
        help="Speech Commands subset to inspect.",
    )
    parser.add_argument("--limit", type=int, default=200, help="Maximum real dataset examples to load.")
    parser.add_argument("--no-download", action="store_true", help="Disable torchaudio dataset download.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = SpeechCommandsKWS(
        data_root=args.data_root,
        subset=args.subset,
        limit=args.limit,
        download=not args.no_download,
    )

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty.")

    waveform, target = dataset[0]
    extractor = LogMelExtractor()
    feature = extractor(waveform.unsqueeze(0))
    counts = summarize_labels(dataset)

    print(f"subset: {args.subset}")
    print(f"dataset_root: {Path(args.data_root)}")
    print(f"examples_including_silence: {len(dataset)}")
    print(f"labels: {', '.join(ALL_LABELS)}")
    print(f"sample_waveform_shape: {tuple(waveform.shape)}")
    print(f"sample_label: {ALL_LABELS[target]} ({target})")
    print(f"sample_logmel_shape: {tuple(feature.shape)}")
    print("label_counts:")
    for label in ALL_LABELS:
        print(f"  {label}: {counts[label]}")


if __name__ == "__main__":
    main()
