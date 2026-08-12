# Benchmarking

Start the CPU service, then run the maintained end-to-end benchmark:

```bash
make docker-cpu-up-d
make benchmark-cpu
```

`complexity/benchmark_batch_complexity.py` sends passport, paired ID-card, and
driving-licence requests at multiple sizes. It records latency, throughput, and
the diagnostics that prove model calls received tensors with `N > 1`. Results
go to ignored `complexity/results/`; other benchmark tools write under
`outputs/benchmarks/`.

Recognition-only comparisons are under `scripts/benchmarking/`. The supported
batch benchmark requires an explicit prepared local cache:

```bash
MODEL_DIR=/path/to/models make benchmark-recognition
```

Use at least one warm-up and three measured repeats. Report medians, keep input
images and configuration constant, close unrelated CPU-heavy work, and record
CPU model/thread details. Batch size means the tensor dimension received by a
model, not the number of one-image calls made by a Python loop.

The older `benchmark.py`, `contrast_benchmark.py`, and `final_benchmark.py`
capture historical model/preprocessing selection experiments. They are not
production defaults. GPU benchmark scripts must run only during the V100 task.
