# Model Card

## Overview
- **Model**: tc_resnet
- **Parameters**: 176,940
- **Training Epochs**: 20
- **Best Validation Accuracy**: 96.01%
- **Testing Accuracy**: 95.49%
- **Silence Accuracy**: 100.00%

## Per-Class Testing Accuracy

- down: 89.16%
- go: 97.26%
- left: 93.93%
- no: 93.33%
- off: 93.53%
- on: 91.67%
- right: 94.95%
- silence: 100.00%
- stop: 95.62%
- unknown: 95.53%
- up: 94.12%
- yes: 98.33%

## Latency Benchmark (CPU, batch=1)

| Backend | Mean (ms) | p50 (ms) | p95 (ms) | Size (MB) |
|---|---|---|---|---|
| PyTorch FP32 | 1.16 | 1.01 | 2.16 | 0.70 |
| ONNX FP32 | 0.54 | 0.33 | 1.68 | 0.68 |
| ONNX INT8 | 4.48 | 3.66 | 8.41 | 0.19 |

## Streaming Inference

- **Target**: yes
- **Threshold**: 0.8
- **Triggered**: True
- **Trigger Time (sec)**: 1.9

## Notes
- All metrics are measured on WSL2 CPU unless otherwise stated.
- INT8 quantization reduced model size but increased latency due to small model size (177K params) and CPU QDQ overhead.