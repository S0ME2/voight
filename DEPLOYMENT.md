# Docker deployment

Copy `.env.example` to `.env`. Set `COMPOSE_PROFILES=cpu`, `RUNTIME_TARGET=cpu`, and `OCR_DEVICE=cpu` for local work; run `make docker-cpu-build`, `make docker-cpu-run`, or `make docker-cpu-test`.

## Local operations

`make docker-up` runs the profile selected in `.env` in the foreground;
`make docker-up-d` does the same in the background. Use `make docker-down` to
stop it, or `make docker-down-v` to also remove the persistent artifact volume.
`make docker-logs`, `make docker-ps`, and `make docker-restart` inspect or
manage the running service. `make docker-shell` opens `sh` inside the CPU app
container. `make docker-logs-copy` copies `/app/logs` to the ignored local
`logs-from-container/` directory. `make docker-logs-clean` deletes every saved
artifact inside `/app/logs`; copy any artifacts you need first.

`requirements/cpu.lock` is the complete CPU install set generated from `uv.lock`; it deliberately replaces the Linux-only transitive GPU ONNX Runtime with the pinned CPU package. The Docker `cpu` and `gpu` targets populate OCR, MRZ, and driving-licence assets during the image build. Runtime images set `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`, so startup never contacts a model source. Persisted artifact logs use the `voight-artifacts` named volume.

On the V100 server only, set all three selectors to `gpu` and run `make docker-gpu-build` followed by `make docker-gpu-test`. The GPU requirements are pinned to `paddlepaddle-gpu==3.2.2` from Paddle's CUDA 11.8 index. Compose reserves one NVIDIA GPU, starts one Uvicorn worker, and uses bounded queue and micro-batch defaults; do not run these commands on a local laptop.
