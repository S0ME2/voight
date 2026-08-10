# Validation runbook

Run the laptop-safe baseline from the repository root:

```bash
make local-validation
python -m json.tool outputs/local-validation-baseline.json
```

The JSON schema has `extraction.baseline`, `extraction.synthetic_robustness`,
and `batch.rows`.  `exact_matches` is a field count for the one annotated
passport or ID-card pair using controlled OCR tokens; it is not an accuracy
percentage.  Synthetic transforms test profile geometry only.  Batch rows show
both single-item and true-batch timings, CPU peak RSS, and each model call's
actual tensor size.

For the V100 server only, preserve that schema and use these commands:

```bash
cp .env.example .env
# Set COMPOSE_PROFILES=gpu, RUNTIME_TARGET=gpu, and GPU_ID=0 in .env.
make docker-gpu-build
make docker-gpu-test
docker compose --env-file .env --profile gpu up -d gpu
curl --fail http://127.0.0.1:${VOIGHT_PORT:-8000}/v1/health/ready
docker compose --env-file .env --profile gpu logs gpu
```

Run the same local-validation command from the checked-out server worktree to
write a comparison baseline.  It never installs or invokes a GPU runtime; use
the GPU service logs and the existing batch diagnostics for real-model V100
measurements in Task 010.
