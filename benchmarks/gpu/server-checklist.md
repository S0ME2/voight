# V100 server checklist

Transfer the repository, `dataset/`, and any prepared CPU comparison artifact.
Do not transfer local model caches unless approved for the server.

Required environment:

```bash
export RUNTIME_TARGET=gpu
export GPU_ID=0
export VOIGHT_GPU_BENCHMARK_HOST=1
export TEXT_RECOGNITION_PROCESSES=1
```

The execution guard defaults to the deployment V100 and driver listed below.
For another GPU, set `VOIGHT_GPU_MODEL` and `VOIGHT_GPU_DRIVER_VERSION` to the
identity reported by `nvidia-smi` before running the benchmark.

Confirm the host is Tesla V100-PCIE-32GB with driver 535.309.01 and host CUDA
capability 12.2, NVIDIA Docker support, free disk, and an unused benchmark
port. Keep other
workloads running and agree on a VRAM ceiling before sweeping.

```bash
nvidia-smi
docker info
docker build --target gpu -t voight:gpu .
uv run --no-sync python benchmarks/gpu/gpu_benchmark.py --plan --mode baseline
uv run --no-sync python benchmarks/gpu/gpu_benchmark.py --execute --mode smoke --port 8090
uv run --no-sync python benchmarks/gpu/gpu_benchmark.py --execute --mode baseline --port 8090
```

Selected sweeps:

```bash
uv run --no-sync python benchmarks/gpu/gpu_benchmark.py --execute --experiment localization-batch --values 1,2,4,8,16,32
uv run --no-sync python benchmarks/gpu/gpu_benchmark.py --execute --experiment detection-batch --values 1,2,4,8,16,32
uv run --no-sync python benchmarks/gpu/gpu_benchmark.py --execute --experiment recognition-batch --values 1,2,4,8,16,32,64
uv run --no-sync python benchmarks/gpu/gpu_benchmark.py --execute --experiment mrz-batch --values 1,2,4,8,16,32
uv run --no-sync python benchmarks/gpu/gpu_benchmark.py --execute --experiment precision --values fp32,fp16
uv run --no-sync python benchmarks/gpu/gpu_benchmark.py --execute --experiment recognition-packing
uv run --no-sync python benchmarks/gpu/gpu_benchmark.py --execute --experiment detector-resolution --values 100,90,80,70,60,50
uv run --no-sync python benchmarks/gpu/gpu_benchmark.py --execute --experiment concurrency --values 1,2,4,8
```

The runner stops/removes its own container after each configuration and records
`lifecycle.jsonl`. For an emergency stop, use the exact name it printed:

```bash
docker ps --filter name=voight-gpu-bench-
docker stop <exact-container-name>
docker rm -f <exact-container-name>
```

Archive the complete timestamped directory, especially `system.json`,
`experiment.json`, `configs.json`, `raw_results.jsonl`, `raw_results.csv`,
`comparison.csv`, `semantic_differences.json`, `lifecycle.jsonl`,
`gpu_samples.csv`, readiness JSON files, and plots. Send those plus the matching
CPU artifact directory back for `compare_cpu_gpu.py`.
