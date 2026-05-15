# AGENTS.md

## Project Name

Mini_Keywords_Arouse: Lightweight Keyword Spoting System for Edge Inference

## Project Goal

This repository implements a minimal but complete keyword spotting system based on Google Speech Commands V2.

The project is designed for resume and interview demonstration. It should emphasize:

- audio waveform processing
- log-Mel feature extraction
- lightweight PyTorch model training
- streaming keyword spotting simulation
- ONNX export
- ONNX Runtime inference benchmark
- INT8 quantization
- reproducible metrics and clean engineering structure

The goal is not to achieve SOTA accuracy. The goal is to build a clear, runnable, measurable, and resume-ready engineering project.

---

## User Development Environment

The user develops locally in Windows WSL2.

Assume the project is edited and tested inside WSL2, not on a remote server.

Preferred environment:

- OS: Ubuntu under WSL2
- Python: 3.10
- Package manager: uv venv preferred, use uv pip install to set up packages enviroment
- Framework: PyTorch + torchaudio
- Inference: ONNX + ONNX Runtime
- GPU: limited MEM, consider the training cost in 24 GB memory
- Shell: zsh or bash

Do not assume CUDA is available.

All smoke tests and benchmarks must be able to run on CPU.

---

## Core Principle

Keep the project simple, runnable, and measurable.

Do not over-engineer.

Avoid unnecessary frameworks such as:

- PyTorch Lightning
- Hydra
- W&B
- MLflow
- distributed training
- Docker
- web UI
- real microphone streaming
- complex DCASE-style sound event detection

Prefer:

- plain PyTorch
- argparse
- simple YAML config
- JSON metric outputs
- small smoke tests
- clean README
- reproducible commands

---

## Required Repository Structure

Create or maintain the following structure:

```text
Mini_Keywords_Arouse/
├── AGENTS.md
├── README.md
├── requirements.txt
├── configs/
│   └── mka_smallcnn.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── speech_commands.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── logmel.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── small_cnn.py
│   │   └── tc_resnet.py
│   ├── train.py
│   ├── eval.py
│   ├── export_onnx.py
│   ├── quantize_onnx.py
│   ├── benchmark_latency.py
│   └── stream_infer.py
├── scripts/
│   ├── check_data.py
│   ├── train_smoke.sh
│   ├── train_full.sh
│   ├── export_and_bench.sh
│   └── make_report.py
├── outputs/
│   ├── metrics.json
│   ├── latency.json
│   ├── stream_demo.json
│   └── model_card.md
└── docs/
    └── resume_bullets.md
```

If a file or directory does not exist, create it.

---

## Dataset

Use Google Speech Commands V2 through torchaudio:

```python
torchaudio.datasets.SPEECHCOMMANDS(
    root=data_root,
    url="speech_commands_v0.02",
    download=True,
    subset="training" | "validation" | "testing"
)
```

The default task should be a 12-class keyword spotting setup:

```text
yes, no, up, down, left, right, on, off, stop, go, unknown, silence
```

Rules:

1. The 10 target words are:

   * yes
   * no
   * up
   * down
   * left
   * right
   * on
   * off
   * stop
   * go

2. Words outside the 10 target words should be mapped to `unknown`.

3. Silence samples can be synthesized from:

   * zero waveform
   * very low-amplitude random noise
   * background noise clips if available

4. All audio should be converted to:

   * mono
   * 16 kHz
   * fixed 1 second length

5. If a waveform is shorter than 1 second, pad it.

6. If a waveform is longer than 1 second, crop it.

---

## Feature Extraction

Implement log-Mel spectrogram extraction.

Default settings:

```yaml
sample_rate: 16000
n_fft: 400
hop_length: 160
win_length: 400
n_mels: 40
f_min: 20
f_max: 7600
```

Expected feature shape should be close to:

```text
[B, 1, 40, T]
```

where `T` is usually around 101 for 1-second audio with hop length 160.

Feature extraction should be implemented in:

```text
src/features/logmel.py
```

---

## Model Requirements

Implement at least one lightweight baseline model:

```text
SmallCNN
```

Optional second model:

```text
TCResNet
```

The model should:

* accept input shape `[B, 1, n_mels, time]`
* output logits with shape `[B, num_classes]`
* be small enough for CPU inference benchmark
* expose a function to count parameters

Do not implement large Transformer models unless explicitly requested.

---

## Training Requirements

`src/train.py` must support:

```bash
python -m src.train \
  --config configs/mka_smallcnn.yaml \
  --limit 2000 \
  --epochs 2
```

Required behavior:

1. Load Speech Commands dataset.
2. Build train and validation DataLoaders.
3. Train SmallCNN.
4. Print training loss and validation accuracy.
5. Save best checkpoint to:

```text
outputs/best.pt
```

6. Save metrics to:

```text
outputs/metrics.json
```

`metrics.json` should include at least:

```json
{
  "train_loss": [],
  "val_accuracy": [],
  "best_val_accuracy": null,
  "num_parameters": null,
  "epochs": null,
  "limit": null
}
```

Do not fabricate metrics.

If a metric is not available, write `null`.

---

## Evaluation Requirements

`src/eval.py` must support:

```bash
python -m src.eval \
  --ckpt outputs/best.pt \
  --limit 1000
```

Required outputs:

1. Overall test accuracy.
2. Per-class accuracy.
3. Confusion matrix saved as CSV.
4. Updated evaluation results in:

```text
outputs/metrics.json
```

---

## ONNX Export Requirements

`src/export_onnx.py` must support:

```bash
python -m src.export_onnx \
  --ckpt outputs/best.pt \
  --out outputs/model.onnx
```

Requirements:

1. Export the trained PyTorch model to ONNX.
2. Use a fixed dummy input shape, for example:

```text
[1, 1, 40, 101]
```

3. Validate that ONNX Runtime can load the exported model.
4. Save the ONNX model to:

```text
outputs/model.onnx
```

---

## Quantization Requirements

`src/quantize_onnx.py` must support:

```bash
python -m src.quantize_onnx \
  --input outputs/model.onnx \
  --output outputs/model_int8.onnx
```

Requirements:

1. Use ONNX Runtime dynamic quantization.
2. Save the quantized model to:

```text
outputs/model_int8.onnx
```

3. Do not claim speedup unless benchmarked.

---

## Latency Benchmark Requirements

`src/benchmark_latency.py` must support:

```bash
python -m src.benchmark_latency \
  --ckpt outputs/best.pt \
  --onnx outputs/model.onnx \
  --onnx-int8 outputs/model_int8.onnx
```

Benchmark targets:

1. PyTorch FP32 CPU
2. ONNX Runtime FP32 CPU
3. ONNX Runtime INT8 CPU

Benchmark protocol:

* batch size: 1
* warmup: 50 iterations
* repeat: 500 iterations
* report:

  * mean latency
  * p50 latency
  * p95 latency
  * model size

Save results to:

```text
outputs/latency.json
```

Example JSON structure:

```json
{
  "pytorch_fp32": {
    "mean_ms": null,
    "p50_ms": null,
    "p95_ms": null,
    "model_size_mb": null
  },
  "onnx_fp32": {
    "mean_ms": null,
    "p50_ms": null,
    "p95_ms": null,
    "model_size_mb": null
  },
  "onnx_int8": {
    "mean_ms": null,
    "p50_ms": null,
    "p95_ms": null,
    "model_size_mb": null
  }
}
```

Never fabricate latency numbers.

Only write measured results.

---

## Streaming Inference Requirements

`src/stream_infer.py` must simulate streaming keyword spotting.

Required command:

```bash
python -m src.stream_infer \
  --ckpt outputs/best.pt \
  --target yes \
  --threshold 0.8
```

Streaming logic:

1. Simulate continuous audio with sliding windows.
2. Default window size:

```text
1.0 second
```

3. Default hop size:

```text
0.1 second
```

4. For each window:

   * extract log-Mel feature
   * run model
   * get target keyword probability

5. Implement confidence smoothing:

   * moving average over recent N windows
   * or exponential moving average

6. Trigger rule:

   * trigger if target probability exceeds threshold for K consecutive windows

Save demo output to:

```text
outputs/stream_demo.json
```

Output should include:

```json
{
  "target": "yes",
  "threshold": 0.8,
  "triggered": true,
  "trigger_time_sec": null,
  "timeline": []
}
```

---

## Reporting Requirements

`scripts/make_report.py` should read:

```text
outputs/metrics.json
outputs/latency.json
outputs/stream_demo.json
```

and generate:

```text
outputs/model_card.md
docs/resume_bullets.md
```

The resume bullets must follow this structure:

```text
Scenario / Problem → Method / System Design → Quantitative Result
```

Do not fabricate numbers.

If a value is missing, use:

```text
TODO
```

Example resume bullet format:

```text
- Built a lightweight keyword spotting system on Google Speech Commands V2, including waveform loading, 16 kHz resampling, log-Mel feature extraction, SpecAugment, SmallCNN training, and evaluation pipeline; achieved TODO% test accuracy with TODO parameters.
- Implemented PyTorch → ONNX → ONNX Runtime deployment and INT8 dynamic quantization; reduced model size from TODO MB to TODO MB and measured CPU p95 latency of TODO ms at batch size 1.
- Designed a sliding-window streaming inference strategy with confidence smoothing and threshold-based triggering, simulating low-latency keyword detection on continuous audio streams.
```

---

## Required Commands

The following commands must work in WSL2:

```bash
# Create environment
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Check dataset
python scripts/check_data.py --limit 2000

# Smoke training
python -m src.train --config configs/mka_smallcnn.yaml --limit 2000 --epochs 2

# Evaluation
python -m src.eval --ckpt outputs/best.pt --limit 1000

# Export ONNX
python -m src.export_onnx --ckpt outputs/best.pt --out outputs/model.onnx

# Quantize ONNX
python -m src.quantize_onnx --input outputs/model.onnx --output outputs/model_int8.onnx

# Benchmark latency
python -m src.benchmark_latency \
  --ckpt outputs/best.pt \
  --onnx outputs/model.onnx \
  --onnx-int8 outputs/model_int8.onnx

# Streaming inference demo
python -m src.stream_infer --ckpt outputs/best.pt --target yes --threshold 0.8

# Generate report and resume bullets
python scripts/make_report.py
```

If `uv` is unavailable, fall back to:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## requirements.txt Guidance

Use minimal dependencies:

```text
torch
torchaudio
numpy
scipy
scikit-learn
pyyaml
onnx
onnxruntime
onnxruntime-tools
pandas
tqdm
```

Avoid adding unnecessary dependencies.

If a package causes installation issues in WSL2, choose the simplest stable alternative.

---

## WSL and Network Notes

The dataset may be downloaded by torchaudio.

If download fails because of network or proxy issues:

1. Do not rewrite the whole dataset logic.
2. Add a clear `--data-root` argument.
3. Allow the user to manually place the extracted Speech Commands folder under:

```text
data/SpeechCommands/
```

4. Document the manual download fallback in README.

Do not hard-code proxy settings in Python code.

Do not modify shell startup files such as:

```text
~/.zshrc
~/.bashrc
```

unless explicitly requested.

---

## Coding Style

* Keep code readable.
* Use type hints where helpful.
* Use clear function names.
* Keep each file focused on one responsibility.
* Use `argparse` for CLI arguments.
* Use `pathlib.Path` for paths.
* Save all generated outputs under `outputs/`.
* Do not write large temporary files outside the project directory.
* Do not assume absolute paths.
* Do not silently ignore exceptions.
* Print useful error messages.

---

## Testing Rules

After each implementation step, run the smallest relevant smoke test.

Examples:

```bash
python scripts/check_data.py --limit 100
python -m src.train --config configs/mka_smallcnn.yaml --limit 200 --epochs 1
python -m src.eval --ckpt outputs/best.pt --limit 100
```

Fix all errors before moving to the next step.

---

## Resume Orientation

Every implemented feature should support at least one resume claim:

1. Built an audio data processing pipeline.
2. Trained a lightweight keyword spotting model.
3. Implemented streaming keyword detection.
4. Exported PyTorch model to ONNX.
5. Applied INT8 quantization.
6. Benchmarked CPU inference latency.
7. Produced reproducible metrics and documentation.

The final README and `docs/resume_bullets.md` are part of the deliverable, not optional extras.

---

## Important Restrictions

Never fabricate:

* accuracy
* latency
* model size
* throughput
* false alarm rate
* training speed
* GPU utilization

If a number is not measured, write `TODO`.

Do not claim real edge-device deployment unless actually tested on an edge device.

It is acceptable to say:

```text
CPU batch=1 benchmark on WSL2
```

It is not acceptable to say:

```text
deployed on Raspberry Pi
```

unless actually tested there.

---

## Final Deliverable

The final project should be runnable, measurable, and easy to inspect.

At completion, the repository should contain:

1. Working training script.
2. Working evaluation script.
3. Working ONNX export.
4. Working INT8 quantization.
5. Working latency benchmark.
6. Working streaming inference simulation.
7. README with commands and result tables.
8. Resume bullets generated from real metrics.

The project should be suitable for demonstrating applied deep learning engineering ability in audio, speech, edge AI, and one-dimensional time-series modeling interviews.