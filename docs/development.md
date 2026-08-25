# Development

## Local CPU setup

Python 3.12 and `uv` are required:

```bash
make install
make test
make check
make run-dev
```

The local run targets deliberately ignore Docker's `/opt/voight/models` path
and let the pinned libraries use the current user's cache. Docker remains the
reproducible path that provisions every model without host preparation.

Never install or execute `requirements/gpu.txt` locally. The complete local
suite is CPU-safe; its real-model batching test uses a local prepared Paddle
cache when present and skips otherwise.

Historical exploration scripts are preserved under `archive/scripts-experiments/`.
They are optional and not part of the normal environment. Use their separate
`requirements/experiments.txt` only in an isolated CPU environment; never mix
it into the production or GPU images.

## Adding a supported document layout

1. Add the source layout to `config/annotation_layouts.json`.
2. Run the guided geometry annotator documented in [dataset.md](dataset.md).
3. Promote reusable geometry into `config/documents/<layout>/profile.json`.
4. Add the small document-specific parser/reconciliation behavior under
   `app/documents/`.
5. Route regions through the existing `ProfileBatchRunner`; do not create a
   parallel sequential pipeline.
6. Add profile, controlled-truth, API, and batch tests.

Only the supplied Uzbekistan layouts are supported today. New layouts require
new evidence and an explicit product decision.

## Adding a model adapter

Implement the relevant contract in `app/inference/contracts.py`, construct the
heavy object in `app/models.py`, preserve actual tensor batching, expose the
selection through validated settings, and add batch-size/error-isolation tests.
Do not add a factory hierarchy for one implementation.

Generated files belong under `outputs/` or `logs/`. Local sensitive images and
ground truth belong under ignored `dataset/`, never beside production code.
