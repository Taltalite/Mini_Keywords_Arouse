# Mini Keywords Arouse

Lightweight keyword spotting baseline for Google Speech Commands V2. The first
stage focuses on a minimal CPU-runnable PyTorch pipeline:

- load Speech Commands V2 with `torchaudio`
- map audio to 12 classes: `yes`, `no`, `up`, `down`, `left`, `right`, `on`,
  `off`, `stop`, `go`, `unknown`, `silence`
- normalize waveforms to mono, 16 kHz, fixed 1 second length
- extract log-Mel features
- train and evaluate a compact `SmallCNN`
- save reproducible metrics under `outputs/`

ONNX export, quantization, latency benchmark, and streaming inference are planned
for the next phase after the PyTorch training and validation loop is stable.

## Environment

Recommended WSL2 setup:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Fallback if `uv` is unavailable:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CPU execution is the default. CUDA is not required.

Recent `torchaudio` builds may route WAV decoding through `torchcodec`, which
requires FFmpeg shared libraries. The dataset wrapper first uses the public
`SPEECHCOMMANDS.__getitem__` interface and falls back to public
`SPEECHCOMMANDS.get_metadata()` plus `scipy.io.wavfile` if the local WSL2
environment lacks FFmpeg.

## Training Pipeline

Minimal closed-loop CPU pipeline:

```bash
python scripts/check_data.py --limit 200
python -m src.train --config configs/mka_smallcnn.yaml --limit 200 --epochs 1
python -m src.eval --ckpt outputs/best.pt --subset validation --limit 100
```

This checks data loading and feature shape, trains `SmallCNN`, saves
`outputs/best.pt`, writes `outputs/metrics.json`, and produces
`outputs/confusion_matrix_validation.csv`.

## ONNX Export

Export the trained `SmallCNN` checkpoint and validate the exported model with
ONNX Runtime:

```bash
python -m src.export_onnx \
  --ckpt outputs/best.pt \
  --config configs/mka_smallcnn.yaml \
  --out outputs/model.onnx
```

The same command is available as:

```bash
bash scripts/export_onnx.sh
```

The export step infers the actual log-Mel input shape with the feature
extractor, writes `outputs/model.onnx`, runs one ONNX Runtime CPU dummy
inference, and saves real numerical comparison results to
`outputs/onnx_check.json`.

## First-Stage Smoke Tests

Check dataset loading and feature shape:

```bash
python scripts/check_data.py --limit 200
```

Train one small CPU epoch:

```bash
python -m src.train --config configs/mka_smallcnn.yaml --limit 200 --epochs 1
```

Validate the saved checkpoint on the Speech Commands validation subset:

```bash
python -m src.eval --ckpt outputs/best.pt --subset validation --limit 100
```

Evaluate the saved checkpoint on the Speech Commands testing subset:

```bash
python -m src.eval --ckpt outputs/best.pt --limit 100
```

Expected generated files:

```text
outputs/best.pt
outputs/metrics.json
outputs/confusion_matrix_validation.csv
outputs/confusion_matrix.csv
```

The same closed-loop smoke flow is available as:

```bash
bash scripts/train_smoke.sh
```

## Dataset

The project uses:

```python
torchaudio.datasets.SPEECHCOMMANDS(
    root=data_root,
    url="speech_commands_v0.02",
    download=True,
    subset="training" | "validation" | "testing",
)
```

Default data root:

```text
data/SpeechCommands
```

If automatic download fails because of network or proxy issues, manually place
the extracted Speech Commands V2 data under the same root and rerun commands with
`--no-download`.

## Current Project Layout

```text
Mini_Keywords_Arouse/
├── README.md
├── requirements.txt
├── configs/
│   └── mka_smallcnn.yaml
├── scripts/
│   ├── check_data.py
│   ├── train_smoke.sh
│   └── train_full.sh
├── src/
│   ├── data/
│   │   └── speech_commands.py
│   ├── features/
│   │   └── logmel.py
│   ├── models/
│   │   └── small_cnn.py
│   ├── train.py
│   └── eval.py
└── outputs/
```

## Notes

Metrics in `outputs/metrics.json` are written only from actual training and
evaluation runs. Missing or future-stage results are not fabricated.
