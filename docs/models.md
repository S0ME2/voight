# Models

## Defaults

| Role | Backend/model | Status |
|---|---|---|
| Document localizer | DocAligner `fastvit_sa24` | default |
| MRZ localizer | MRZScanner detection `20250222` | default |
| Text detector | Paddle `PP-OCRv6_medium_det` | default |
| Text recognizer | Paddle `latin_PP-OCRv5_mobile_rec` | default |
| MRZ recognizer | generic Paddle recognizer | default |
| Specialized MRZ recognizer | MRZScanner `20250221` | experimental |

PaddleOCR, PaddlePaddle, DocAligner, and MRZScanner are Apache-licensed. The
optional historical FastMRZ experiment is AGPL-licensed and is excluded from
production dependencies.

## Docker provisioning

`make docker-cpu-build` runs `scripts/models/prepare.py` during the image build.
Paddle weights are downloaded into `/opt/voight/models/official_models`; the
DocAligner and MRZScanner ONNX files are supplied by their pinned Python wheels.
A manifest is written to `/opt/voight/models/voight-models.json`:

```bash
make models-info
```

No model volume or host `~/.paddlex` directory is used. Docker reuses the model
layer while its inputs are unchanged. The first build requires internet access;
an already-built image starts and serves requests without downloading models.

Remove the local image with `docker image rm voight:cpu`. Force dependency and
weight downloads again with `make models-rebuild-cpu`. Docker may retain an
unused build-cache layer; broad builder-cache pruning is intentionally not part
of the Makefile because it affects unrelated projects.

## Swapping or supplying models

Model names and backends are selected through `.env`. A selected Paddle model
must exist under `MODEL_DIR/official_models/<model-name>`. The default image
contains only its documented models; changing a name at runtime without
supplying its files fails readiness.

For custom compatible Paddle weights, mount a prepared root read-only:

```bash
docker run --rm -p 8000:8000 \
  -v "$PWD/my-models:/models:ro" \
  -e MODEL_DIR=/models \
  -e RUNTIME_TARGET=cpu \
  voight:cpu
```

The custom directory must contain `official_models/<configured-name>/` in the
layout expected by PaddleOCR. A derived image using the same path is preferable
for reproducible deployment. Model adapters may be added through `Models` only
when their contract behavior and batch semantics are tested.
