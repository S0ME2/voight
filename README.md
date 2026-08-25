# Voight OCR

Voight is a FastAPI service for profile-driven OCR of three supported document
layouts: the supplied Uzbekistan passport, the supplied Uzbekistan ID card
(front and back together), and the existing driving licence. It performs
document localization, canonicalization, ROI filtering, batched PaddleOCR
recognition, parsing, validation, and structured confidence reporting.

The supplied profiles represent one layout/example per identity-document type.
They are not a claim of general OCR accuracy.

## Quick start: CPU Docker

Requirements: Docker with Compose v2 and internet access for the first build.

```bash
git clone <repository-url> voight
cd voight
cp .env.example .env
make docker-cpu-build
make docker-cpu-up-d
curl --fail http://127.0.0.1:8000/v1/health/ready
```

The build prepares the pinned Paddle model assets inside the image. It does not
read or mount `~/.paddlex` from the host. Stop the service with:

```bash
make docker-cpu-down
```

## API example

```bash
curl --fail -F image=@passport.jpg \
  http://127.0.0.1:8000/v1/ocr/passport
```

Swagger UI is available at <http://127.0.0.1:8000/docs>. Copy-paste examples
for every route and batch layout are in [docs/api.md](docs/api.md).

## Development

Local development is CPU-only:

```bash
make install
make test
make run-dev
```

Use `make help` for all supported Docker, model, dataset, validation, and
benchmark commands. Generated artifacts go to ignored `logs/` and `outputs/`
directories; private local datasets belong in ignored `dataset/`.

## Project map

```text
app/               production API, document pipelines, and inference adapters
config/            versioned document profiles and ROI configuration
scripts/           dataset, validation, and model tools
benchmarks/        maintained and historical benchmark tooling
archive/           preserved early exploration prototypes
tests/             CPU-safe unit, contract, batching, and integration tests
annotation_input/  authorized committed profile-validation fixtures
annotations/       authorized committed annotation state and truth
docs/              architecture and operating documentation
dataset/           ignored local sensitive data
logs/, outputs/    ignored generated artifacts
```

## Documentation

- [Architecture](docs/architecture.md)
- [API](docs/api.md)
- [Configuration](docs/configuration.md)
- [Models](docs/models.md)
- [CPU/GPU deployment](docs/deployment.md)
- [Development](docs/development.md)
- [Benchmarking](docs/benchmarking.md)
- [GPU benchmark suite](benchmarks/gpu/README.md)
- [Datasets and annotation](docs/dataset.md)
- [Legacy API migration](docs/migration.md)

GPU manifests and the staged benchmark suite are provided for the Tesla V100
server, but GPU inference has not been validated yet. Do not run GPU targets or
the benchmark executor on a CPU development machine.
