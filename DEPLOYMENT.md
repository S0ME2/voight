# Docker deployment

Copy `.env.example` to `.env`. Set `COMPOSE_PROFILES=cpu` and `RUNTIME_TARGET=cpu` for local work; run `make docker-cpu-build`, `make docker-cpu-run`, or `make docker-cpu-test`. `OCR_DEVICE` remains accepted as a deprecated fallback when `RUNTIME_TARGET` is absent; do not set both in new deployments.

## Local operations

`make docker-up` runs the profile selected in `.env` in the foreground;
`make docker-up-d` does the same in the background. Use `make docker-down` to
stop it, or `make docker-down-v` to also remove the persistent artifact volume.
`make docker-logs`, `make docker-ps`, and `make docker-restart` inspect or
manage the running service. `make docker-shell` opens `sh` inside the CPU app
container. `make docker-logs-copy` copies `/app/logs` to the ignored local
`logs-from-container/` directory. `make docker-logs-clean` deletes every saved
artifact inside `/app/logs`; copy any artifacts you need first.

`requirements/cpu.lock` is the complete CPU production install set and is installed with `--no-deps`; it deliberately replaces DocSaid/Capybara's Linux-only transitive GPU ONNX Runtime with the pinned CPU package. The ordinary project and `requirements/cpu.txt` remain CPU-safe and omit the two DocSaid wrappers because their published Linux metadata hard-depends on `onnxruntime-gpu`; the CPU Docker lock supplies those wrappers safely. The Docker `cpu` and `gpu` targets populate OCR, MRZ, and driving-licence assets during the image build. Runtime images set `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`, so startup never contacts a model source. Persisted artifact logs use the `voight-artifacts` named volume.

`REQUEST_QUEUE_LIMIT` is the maximum number of admitted `/v1` inference requests, including the one currently executing. One admitted request runs the shared models; later admitted requests are suspended async coroutines, not worker threads. A request beyond that limit receives `503 QUEUE_FULL`. Separate HTTP requests are never combined into a model batch.

Visible-document detection polygons are filtered by the same field-ROI center rule used for field assignment before recognition. MRZ crops deliberately bypass that filter. `/v1` diagnostics expose `line_filter.detected_line_count`, `recognition_candidate_count`, and `filtered_before_recognition_count`, plus per-sample counts.

On the V100 server only, set `COMPOSE_PROFILES=gpu`, `RUNTIME_TARGET=gpu`, and `GPU_ID=0`, then run `make docker-gpu-build` followed by `make docker-gpu-test`. The test loads the deployed models and requires Paddle CUDA support plus ONNX Runtime's `CUDAExecutionProvider`. The GPU requirements pin `paddlepaddle-gpu==3.2.2` from Paddle's CUDA 11.8 index and `onnxruntime-gpu==1.22.0`. Compose reserves one NVIDIA GPU, starts one Uvicorn worker, and uses bounded queue and micro-batch defaults; do not run these commands on a local laptop.

Text-recognition acceleration remains opt-in and GPU-only: `TEXT_RECOGNITION_ENABLE_HPI=false`, `TEXT_RECOGNITION_USE_TENSORRT=false`, and `TEXT_RECOGNITION_PRECISION=fp32` preserve the ordinary Paddle constructor. Valid precision is `fp32` or `fp16`; HPI, TensorRT, and FP16 fail startup on CPU rather than silently changing runtimes.

Model selection is centralized. For example,
`TEXT_RECOGNIZER_MODEL=PP-OCRv6_small_rec` swaps the general recognizer without
pipeline edits when that model is already cached. `TEXT_RECOGNITION_PACKING`
selects `sequential` (default) or the faster but opt-in `aspect-ratio`. `MRZ_RECOGNIZER_BACKEND` is
`generic-paddle` by default; `mrzscanner` is an explicit comparison option and
does not silently fall back. Current MRZScanner 1.0.7 recognition is fixed to
tensor batch size one and did not pass the supplied ID-card validity check.
