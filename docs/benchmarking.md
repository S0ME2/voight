# Benchmarking

Start the CPU service, then run the maintained end-to-end benchmark:

```bash
make docker-cpu-up-d
make benchmark-cpu
```

`benchmarks/maintained/benchmark_batch_complexity.py` sends passport, paired
ID-card, and driving-licence requests at multiple sizes. It records latency,
throughput, and the diagnostics that prove model calls received tensors with
`N > 1`. Results go to ignored `benchmarks/results/`; other benchmark tools
write under `outputs/benchmarks/`.

Maintained recognition-only comparisons are under `benchmarks/maintained/`.
The supported batch benchmark requires an explicit prepared local cache:

```bash
MODEL_DIR=/path/to/models make benchmark-recognition
```

Use at least one warm-up and three measured repeats. Report medians, keep input
images and configuration constant, close unrelated CPU-heavy work, and record
CPU model/thread details. Batch size means the tensor dimension received by a
model, not the number of one-image calls made by a Python loop.

Tools whose purpose is reproducing a completed experiment live in
`benchmarks/historical/`: the older `benchmark.py`, `contrast_benchmark.py`,
and `final_benchmark.py` capture historical model/preprocessing selection
experiments, and the remaining scripts reproduce the individual CPU-selection
experiments (model matrix, batch sweep, thread sweep, detector resolution,
packing, recognizer A/B, MRZ preprocessing reconciliation) behind the locked
configuration in `.codex/DECISIONS.md` D-021. They are not production
defaults. GPU benchmark scripts must run only during the V100 task.

The fixed-crop preprocessing benchmark runs Phases A-D on CPU with the
measured L4-D1-R2 settings and writes raw JSONL, fixed `.npy` crops, compact
CSV tables, finalist payloads, and lifecycle evidence under
`outputs/benchmarks/preprocessing/<timestamp>/`:

```bash
uv run --no-sync python benchmarks/maintained/preprocessing_benchmark.py \
  --model-dir models/benchmark --repeats 3
```

Its benchmark-only environment hooks are unset by default; the production
pipeline therefore keeps its existing preprocessing behavior.

The acceptance-gate checklist for the corrected CPU thread-count benchmark
that locked `CPU_THREADS=4` is recorded in [gates.md](gates.md). Local
experiment reports are kept in the ignored `experiments/` directory and are
indexed, with per-report status and evidence links, by
`experiments/README.md`.

The reusable GPU suite is in [benchmarks/gpu](../benchmarks/gpu/README.md).
Use its `--plan` mode on a CPU laptop; execute mode is guarded for the V100
server and runs one fresh container per configuration.
