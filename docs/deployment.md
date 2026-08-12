# Deployment

## CPU

```bash
cp .env.example .env
make docker-cpu-build
make docker-cpu-up-d
curl --fail http://127.0.0.1:8000/v1/health/ready
make docker-cpu-logs
make docker-cpu-down
```

The image uses pinned CPU Paddle and ONNX Runtime packages, runs one Uvicorn
worker as the unprivileged `voight` user, exposes container port 8000, and has a
readiness health check. `VOIGHT_PORT` controls the host port. The
`voight-artifacts` named volume persists `/app/logs` across container recreation.
Use `make docker-cpu-artifacts-copy` before
`make docker-cpu-artifacts-clean` when saved debugging output is needed.

```bash
docker volume ls
docker volume rm voight_voight-artifacts  # destructive: removes saved artifacts
```

The exact Compose-prefixed volume name can vary with the project directory; use
`docker volume ls` before removal. Models are stored in the image, not this
volume.

## GPU

GPU commands are server-only. Set `RUNTIME_TARGET=gpu` and `GPU_ID` in `.env`:

```bash
make docker-gpu-build
make docker-gpu-test
make docker-gpu-up-d
```

The GPU target pins Paddle GPU for CUDA 11.8 and ONNX Runtime GPU, requests one
NVIDIA GPU, keeps one application worker, and shares the production pipeline.
HPI, TensorRT, and FP16 remain disabled unless explicitly configured. GPU
inference has not yet been verified on the target V100; see the limitation in
the root README and `.codex/tasks/TASK-010-v100-server-validation.md`.

Rollback is `make docker-gpu-down`, set `RUNTIME_TARGET=cpu`, then run the CPU
commands above. Never build, import-test, or run the GPU target on a CPU laptop.

## Offline behavior

The first image build needs internet access for Python packages and Paddle
weights. Docker can rebuild from local layer cache when all required layers are
present. A built image can start and serve offline because runtime source checks
are disabled and readiness verifies the baked assets.
