# Mini Keywords Arouse

Lightweight keyword spotting system for Google Speech Commands V2, designed for
CPU-runnable edge-AI demonstration:

- load Speech Commands V2 with `torchaudio`
- map audio to 12 classes: `yes`, `no`, `up`, `down`, `left`, `right`, `on`,
  `off`, `stop`, `go`, `unknown`, `silence`
- normalize waveforms to mono, 16 kHz, fixed 1 second length
- extract log-Mel features
- train and evaluate compact `SmallCNN` (optional `TCResNet`)
- export to ONNX, apply INT8 dynamic quantization, benchmark CPU latency
- simulate sliding-window streaming keyword detection
- save reproducible metrics and documentation under `outputs/` and `docs/`

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

## Quick Start

Run the full closed-loop pipeline on a small CPU smoke subset:

```bash
python scripts/check_data.py --limit 200
python -m src.train --config configs/mka_smallcnn.yaml --limit 200 --epochs 1
python -m src.eval --ckpt outputs/best.pt --limit 100
python -m src.export_onnx --ckpt outputs/best.pt --out outputs/model.onnx
python -m src.quantize_onnx --input outputs/model.onnx --output outputs/model_int8.onnx
python -m src.benchmark_latency \
  --ckpt outputs/best.pt \
  --onnx outputs/model.onnx \
  --onnx-int8 outputs/model_int8.onnx
python -m src.stream_infer --ckpt outputs/best.pt --target yes --threshold 0.8
python scripts/make_report.py
```

The same export → quantize → benchmark flow is available as:

```bash
bash scripts/export_and_bench.sh
```

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

For a longer run:

```bash
bash scripts/train_full.sh
```

## ONNX Export

Export the trained checkpoint and validate the exported model with ONNX Runtime:

```bash
python -m src.export_onnx \
  --ckpt outputs/best.pt \
  --out outputs/model.onnx
```

`--config` is optional; when omitted, the script reads `feature_config`,
`data_config`, and `model_config` from the checkpoint. Pass `--config` only if
you want to override checkpoint values.

The same command is available as:

```bash
bash scripts/export_onnx.sh
```

The export step infers the actual log-Mel input shape with the feature
extractor, writes `outputs/model.onnx`, runs one ONNX Runtime CPU dummy
inference, and saves real numerical comparison results to
`outputs/onnx_check.json`.

## INT8 Quantization

Apply ONNX Runtime dynamic quantization to the FP32 ONNX model:

```bash
python -m src.quantize_onnx \
  --input outputs/model.onnx \
  --output outputs/model_int8.onnx
```

This produces `outputs/model_int8.onnx` and a verification summary in
`outputs/quantization_check.json`.

## Latency Benchmark

Compare PyTorch FP32, ONNX FP32, and ONNX INT8 CPU latency at batch size 1:

```bash
python -m src.benchmark_latency \
  --ckpt outputs/best.pt \
  --onnx outputs/model.onnx \
  --onnx-int8 outputs/model_int8.onnx
```

Results are saved to `outputs/latency.json`.

## Streaming Inference

Simulate sliding-window keyword detection on continuous audio:

```bash
python -m src.stream_infer \
  --ckpt outputs/best.pt \
  --target yes \
  --threshold 0.8
```

Default window size is 1.0 s and hop size is 0.1 s. The demo applies EMA
confidence smoothing and triggers when the smoothed probability exceeds the
threshold for several consecutive windows. Output is saved to
`outputs/stream_demo.json`.

## Smoke Tests

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

The same closed-loop smoke flow is available as:

```bash
bash scripts/train_smoke.sh
```

## Reporting

Generate the model card and resume bullets from measured outputs:

```bash
python scripts/make_report.py
```

This reads:

- `outputs/metrics.json`
- `outputs/latency.json`
- `outputs/stream_demo.json`

and writes:

- `outputs/model_card.md`
- `docs/resume_bullets.md`

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

## Project Layout

```text
Mini_Keywords_Arouse/
├── README.md
├── requirements.txt
├── configs/
│   └── mka_smallcnn.yaml
├── scripts/
│   ├── check_data.py
│   ├── train_smoke.sh
│   ├── train_full.sh
│   ├── export_onnx.sh
│   ├── export_and_bench.sh
│   └── make_report.py
├── src/
│   ├── data/
│   │   └── speech_commands.py
│   ├── features/
│   │   └── logmel.py
│   ├── models/
│   │   ├── small_cnn.py
│   │   └── tc_resnet.py
│   ├── train.py
│   ├── eval.py
│   ├── export_onnx.py
│   ├── quantize_onnx.py
│   ├── benchmark_latency.py
│   └── stream_infer.py
├── outputs/
└── docs/
```

## Notes

Metrics in `outputs/metrics.json` are written only from actual training and
evaluation runs. Missing or future-stage results are reported as `TODO` rather
than fabricated.
