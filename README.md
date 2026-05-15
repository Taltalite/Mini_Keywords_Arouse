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
for the next phase after smoke tests pass.

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

## First-Stage Smoke Tests

Check dataset loading and feature shape:

```bash
python scripts/check_data.py --limit 200
```

Train one small CPU epoch:

```bash
python -m src.train --config configs/kws_smallcnn.yaml --limit 200 --epochs 1
```

Evaluate the saved checkpoint:

```bash
python -m src.eval --ckpt outputs/best.pt --limit 100
```

Expected generated files:

```text
outputs/best.pt
outputs/metrics.json
outputs/confusion_matrix.csv
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
│   └── kws_smallcnn.yaml
├── scripts/
│   └── check_data.py
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
