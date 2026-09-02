# Repository reorganization — August 2026

This is a historical record of the repository layout cleanup. It documents
where maintained tooling, historical benchmark drivers, and archived
exploration code belong. It is not the product API specification; see the
[root README](../README.md) and [API reference](api.md) for current behavior.

## Resulting structure

```text
app/                    production service
config/                 document profiles and runtime configuration
tests/                  CPU-safe test suite
docs/                   operating documentation and historical records
scripts/dataset/        ground-truth and profile annotation tooling
scripts/models/         model-cache preparation
scripts/validation/     local validation tooling
benchmarks/maintained/  reusable benchmark drivers
benchmarks/historical/  completed-experiment reproduction drivers
benchmarks/gpu/         server-only V100 benchmark suite
archive/                preserved legacy scripts and assets
```

The cleanup kept the supported document behavior intact. The maintained
benchmark entry points now live under `benchmarks/maintained/`, completed
experiment drivers under `benchmarks/historical/`, and old exploration code
under `archive/scripts-experiments/`. Benchmark results use
`benchmarks/results/` or `outputs/benchmarks/`, depending on the driver.

## Tracked path changes

| Former area | Current area | Role |
|---|---|---|
| `complexity/` | `benchmarks/maintained/` | Maintained end-to-end benchmark |
| `scripts/benchmarking/` | `benchmarks/maintained/` | Reusable benchmark tooling |
| `scripts/benchmarking/` | `benchmarks/historical/` | One-off completed experiments |
| Former exploration scripts | `archive/scripts-experiments/` | Preserved prototypes |
| Legacy asset samples | `archive/assets-legacy/` | Preserved duplicate or unused assets |

Reference updates included Make targets, imports between benchmark drivers,
default result locations, compilation coverage, cleanup coverage, and test
imports. Production code remains separate from benchmark and archive code.

## Deliberate boundaries

- `app/` owns request handling, document pipelines, inference coordination,
  schemas, and artifact behavior.
- `config/documents/` owns reusable profile geometry and field definitions.
- `benchmarks/maintained/` contains tools that are still useful for future
  measurements; historical drivers are not production defaults.
- The GPU benchmark suite is planned and checked on CPU, but executes only on
  the specified V100 deployment server.
- Sensitive fixtures, local model caches, and generated artifacts remain
  outside the tracked product source and are not part of this layout change.

## Decisions retained

- No generic `benchmarks/common/` framework was introduced. Existing shared
  helpers are sufficient, and extracting a framework would broaden the change
  without a measured need.
- The existing driving-licence behavior was preserved while the profile
  pipelines were organized.
- The CPU configuration remains the measured baseline: four CPU threads,
  fixed-width recognition packing, and the documented model-stage batch sizes.
- Historical benchmark evidence remains reproducible without making its
  drivers part of the runtime image.

## Validation recorded at the time

- `make check`
- `make test`
- CPU-safe compilation of `app`, `scripts`, `tests`, `benchmarks`, and `archive`
- stale-reference review after each path move
- working-tree review confirming no commit was made by the cleanup

For current development commands, dataset handling, deployment, and benchmark
procedures, use the living documents linked from the [root README](../README.md).
