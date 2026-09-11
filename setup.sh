#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

usage() {
    cat <<'EOF'
Usage: ./setup.sh [cpu|gpu]

Build, test, start, and wait for the Voight service.
  cpu  Safe default for local machines.
  gpu  Server-only Tesla V100 deployment; requires nvidia-smi.

The script creates .env from .env.example when needed. Set ENV_FILE to use
another environment file.
EOF
}

fail() {
    echo "setup: $*" >&2
    exit 1
}

case "${1:-}" in
    ""|cpu|gpu) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

command -v docker >/dev/null 2>&1 || fail "Docker is required"
command -v make >/dev/null 2>&1 || fail "make is required"
docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is required"

ENV_FILE="${ENV_FILE:-.env}"
if [[ "$ENV_FILE" != /* ]]; then
    ENV_FILE="$ROOT_DIR/$ENV_FILE"
fi
if [[ ! -f "$ENV_FILE" ]]; then
    cp .env.example "$ENV_FILE" || fail "could not create $ENV_FILE"
    echo "Created $ENV_FILE from .env.example"
fi

env_value() {
    awk -F= -v name="$1" '$1 == name { sub(/\r$/, "", $2); print $2; exit }' "$ENV_FILE"
}

set_runtime_target() {
    if grep -q '^RUNTIME_TARGET=' "$ENV_FILE"; then
        sed -i "s/^RUNTIME_TARGET=.*/RUNTIME_TARGET=$1/" "$ENV_FILE"
    else
        printf '\nRUNTIME_TARGET=%s\n' "$1" >> "$ENV_FILE"
    fi
}

mode="${1:-cpu}"
mode="${mode:-cpu}"
if [[ "$mode" != cpu && "$mode" != gpu ]]; then
    fail "RUNTIME_TARGET must be cpu or gpu (got: $mode)"
fi
if [[ "$(env_value RUNTIME_TARGET)" != "$mode" ]]; then
    set_runtime_target "$mode"
fi

if [[ "$mode" == gpu ]]; then
    [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || \
        fail "GPU setup requires Linux x86_64"
    command -v nvidia-smi >/dev/null 2>&1 || \
        fail "GPU setup requires nvidia-smi; run this only on the V100 server"
    gpu_id="${GPU_ID:-$(env_value GPU_ID)}"
    gpu_id="${gpu_id:-0}"
    [[ "$gpu_id" =~ ^[0-9]+$ ]] || fail "GPU_ID must be a non-negative integer"
    nvidia-smi -i "$gpu_id" --query-gpu=name,driver_version --format=csv,noheader \
        || fail "GPU_ID=$gpu_id is not available"
fi

make_args=("ENV_FILE=$ENV_FILE")
compose=(docker compose --env-file "$ENV_FILE")
service="$mode"
if [[ "$mode" == gpu ]]; then
    compose+=(--profile gpu)
fi

"${compose[@]}" config --quiet || fail "Compose configuration is invalid"
make "${make_args[@]}" "docker-$mode-build"
make "${make_args[@]}" "docker-$mode-test"
make "${make_args[@]}" "docker-$mode-up-d"

echo "Waiting for $service readiness..."
for _ in $(seq 1 90); do
    if "${compose[@]}" exec -T "$service" python -c \
        'from urllib.request import urlopen; response = urlopen("http://127.0.0.1:8000/v1/health/ready", timeout=2); assert response.status == 200' \
        >/dev/null 2>&1; then
        port="${VOIGHT_PORT:-$(env_value VOIGHT_PORT)}"
        echo "Voight $mode service is ready at http://127.0.0.1:${port:-8000}"
        exit 0
    fi
    sleep 2
done

"${compose[@]}" logs --tail=100 "$service" >&2 || true
fail "service did not become ready within 180 seconds"
