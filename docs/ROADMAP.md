# Mini Keywords Arouse Engineering Roadmap

This roadmap records the current engineering state and the staged path toward a CPU-runnable edge keyword spotting demo. Metrics must come from actual runs under `outputs/`; missing measurements stay as TODO.

## Current State

The project currently has a minimal PyTorch training and validation pipeline:

- Dataset wrapper for Google Speech Commands V2 in `src/data/speech_commands.py`.
- 12-class KWS label mapping: 10 target keywords plus `unknown` and `silence`.
- Mono, 16 kHz, fixed 1 second waveform normalization.
- Log-Mel feature extraction in `src/features/logmel.py`.
- Lightweight `SmallCNN` model in `src/models/small_cnn.py`.
- Training entrypoint in `src/train.py`.
- Evaluation entrypoint in `src/eval.py`.
- Primary config in `configs/mka_smallcnn.yaml`.
- Small CPU smoke scripts under `scripts/`.

Current runnable training command:

```bash
python -m src.train --config configs/mka_smallcnn.yaml --limit 200 --epochs 1
```

Current validation command:

```bash
python -m src.eval --ckpt outputs/best.pt --subset validation --limit 100
```

## Missing Functionality

- ONNX export from `outputs/best.pt`.
- ONNX Runtime model load and inference validation.
- INT8 dynamic quantization.
- CPU latency benchmark for PyTorch, ONNX FP32, and ONNX INT8.
- Streaming keyword spotting simulation.
- Local API demo for one-shot inference.
- Report generation from real metrics.
- Model card and resume bullets.
- Final README result tables.

## Staged Plan

### Stage 1: Training Baseline Stabilization

- Keep `src/train.py`, `src/eval.py`, and `configs/mka_smallcnn.yaml` as the primary training path.
- Maintain `scripts/smoke_train.sh` as the smallest regression check.
- Ensure generated training metrics are written to `outputs/metrics.json`.

Exit check:

```bash
bash scripts/smoke_train.sh
```

### Stage 2: Evaluation Reporting

- Keep validation and testing evaluation explicit through `--subset`.
- Save confusion matrices under `outputs/`.
- Preserve real per-class accuracy values; write `null` when unavailable.

### Stage 3: ONNX Export

- Add `src/export_onnx.py`.
- Load `outputs/best.pt`.
- Export fixed input shape `[1, 1, 40, 101]` to `outputs/model.onnx`.
- Validate model load with ONNX Runtime.

### Stage 4: INT8 Quantization

- Add `src/quantize_onnx.py`.
- Use ONNX Runtime dynamic quantization.
- Save `outputs/model_int8.onnx`.
- Do not claim speedup before benchmark results exist.

### Stage 5: Latency Benchmark

- Add `src/benchmark_latency.py`.
- Measure CPU batch size 1 latency for PyTorch FP32, ONNX FP32, and ONNX INT8.
- Save measured values to `outputs/latency.json`.

### Stage 6: Streaming Simulation

- Add `src/stream_infer.py`.
- Simulate sliding-window keyword spotting with confidence smoothing.
- Save timeline and trigger result to `outputs/stream_demo.json`.

### Stage 7: Local API Demo

- Add a minimal local API only after model export and inference paths are stable.
- Keep dependencies lightweight.
- CPU must remain the default runtime.

### Stage 8: Documentation and Resume Artifacts

- Add `scripts/make_report.py`.
- Generate `outputs/model_card.md`.
- Generate `docs/resume_bullets.md`.
- Use TODO for any missing metrics.

## Constraints

- Do not introduce PyTorch Lightning, Hydra, W&B, MLflow, Docker, or a frontend UI.
- Keep each stage small and independently smoke-tested.
- Do not fabricate accuracy, latency, model size, or throughput.
- Keep WSL2 CPU execution as the default path.
