# Getting started

This is the single first-time setup path for Voight. Choose one runtime before
running commands:

| Choice | Use it when | Start here |
|---|---|---|
| CPU Docker | You want a reproducible service with no host model preparation | `./setup.sh cpu` |
| Local CPU | You are developing Python code and already have a local model cache | `make install` |
| GPU Docker | You are on the designated Tesla V100 deployment server | `./setup.sh gpu` |

Local development and tests are CPU-only. Do not install, import-test, build,
or run the GPU dependency set on a laptop or workstation.

## Recommended: CPU Docker

Requirements: Linux x86_64, Docker Engine with Compose v2, and internet access
for the first image build.

```bash
git clone <repository-url> voight
cd voight
./setup.sh cpu
```

The script creates `.env` if needed, builds the pinned CPU image with its model
assets, runs the CPU test image, starts the service, and waits for readiness.
When it finishes, verify the service:

```bash
curl --fail http://127.0.0.1:8000/v1/health/live
curl --fail http://127.0.0.1:8000/v1/health/ready
curl --fail \
  -F image=@annotation_input/passports/passport.png \
  http://127.0.0.1:8000/v1/ocr/passport
```

The CPU container serves on host port `8000` by default. Set `VOIGHT_PORT` in
`.env` before setup if another port is required. Stop it with:

```bash
make docker-cpu-down
```

The first build downloads packages, model weights, and pinned third-party model
assets. An already-built image can start without downloading models again.

## Local CPU development

Use this path when changing Python code. It does not provision a host model
cache; it uses the current user's prepared Paddle cache when the application
loads real models. The full CPU Docker path above is simpler for a clean clone.

Requirements: Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
make install
make check
make test
make run
```

The local server listens on `http://127.0.0.1:8888`. In another terminal:

```bash
curl --fail http://127.0.0.1:8888/v1/health/live
```

Use `make run-dev` for auto-reload. Real-model tests skip when no local model
cache is selected; no test command downloads models implicitly.

## GPU deployment

Use this path only on the designated Linux x86_64 Tesla V100 deployment server.
The supported host prerequisites and server-only validation checklist are in
[`../benchmarks/gpu/server-checklist.md`](../benchmarks/gpu/server-checklist.md).

```bash
nvidia-smi
git clone <repository-url> voight
cd voight
./setup.sh gpu
```

The script verifies `nvidia-smi`, selects `RUNTIME_TARGET=gpu`, builds the
separate pinned GPU image, runs its readiness check, starts one GPU worker, and
waits for `/v1/health/ready`. GPU execution is server-only and remains pending
runtime validation on the target V100; do not treat a successful image build as
GPU performance or parity validation.

After startup:

```bash
curl --fail http://127.0.0.1:8000/v1/health/ready
make docker-gpu-logs
```

Stop the GPU service with `make docker-gpu-down`. Never run `./setup.sh gpu` on
a CPU-only development machine.

## Where to go next

- [`../README.md`](../README.md) — API choices, request examples, and project limits
- [`deployment.md`](deployment.md) — Compose environment files, artifacts, offline behavior
- [`configuration.md`](configuration.md) — supported environment variables and defaults
- [`models.md`](models.md) — model ownership, provisioning, and cache behavior
- [`api.md`](api.md) — complete route contracts and examples
- [`development.md`](development.md) — adding document layouts or model adapters
- [`architecture.md`](architecture.md) — processing stages and module responsibilities

Only the supplied Uzbekistan passport, Uzbekistan ID-card, and driving-licence
layouts are supported. An ID card is always one logical document containing a
front and back image. Uploaded documents may be written to `LOG_DIR` when
`LOGGING=true`; use authorized data only.
