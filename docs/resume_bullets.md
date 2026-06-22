# Resume Bullets

- Built a lightweight keyword spotting system on Google Speech Commands V2, including waveform loading, 16 kHz resampling, log-Mel feature extraction, SmallCNN training, and evaluation pipeline; achieved 5.45% test accuracy with 24188 parameters.

- Implemented PyTorch → ONNX → ONNX Runtime deployment and INT8 dynamic quantization; reduced model size from 0.0937 MB to 0.0301 MB and measured CPU p95 latency of 0.6032 ms at batch size 1.

- Designed a sliding-window streaming inference strategy with confidence smoothing and threshold-based triggering, simulating low-latency keyword detection on continuous audio streams.
