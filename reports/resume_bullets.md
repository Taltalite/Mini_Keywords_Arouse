# Resume Bullets

## Keyword Spotting System

- Built a lightweight keyword spotting system on Google Speech Commands V2, including waveform loading, 16 kHz resampling, log-Mel feature extraction, tc_resnet training, and evaluation pipeline; achieved 95.49% test accuracy with 176,940 parameters.
- Implemented PyTorch → ONNX → ONNX Runtime deployment and INT8 dynamic quantization; reduced model size from 0.68 MB to 0.19 MB and measured CPU p95 latency of 1.68 ms at batch size 1.
- Designed a sliding-window streaming inference strategy with confidence smoothing and threshold-based triggering, simulating low-latency keyword detection on continuous audio streams; silence classification accuracy reached 100.00%.
- Developed a real-time microphone demo prototype (`src/mic_demo.py`) for Windows, demonstrating end-to-end feasibility from audio capture to model inference.

## Quantitative Summary

| Metric | Value |
|---|---|
| Testing Accuracy | 95.49% |
| Best Validation Accuracy | 96.01% |
| Parameters | 176,940 |
| ONNX FP32 p95 Latency | 1.68 ms |
| ONNX INT8 Model Size | 0.19 MB |
| Streaming Trigger Latency | 1.9 sec |