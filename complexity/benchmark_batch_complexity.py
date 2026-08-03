"""
Benchmark the three OCR batch endpoints with batch sizes from 1 to 60.

Expected project layout:

project/
├── app/
├── complexity/
│   ├── benchmark_batch_complexity.py
│   ├── realidcard.png
│   ├── uzpassport.png
│   └── test_license.jpg
└── logs/

The input paths can be overridden with command-line arguments.

Outputs are written to the directory containing this script:
- benchmark_results.csv
- benchmark_results.json
- benchmark_summary.json
- id_card_batch_time.png
- passport_batch_time.png
- driving_license_batch_time.png
- all_batch_times.png
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import requests


@dataclass(frozen=True)
class EndpointConfig:
    key: str
    label: str
    path: str
    image_path: Path
    log_operation: str


@dataclass
class Measurement:
    endpoint: str
    endpoint_label: str
    endpoint_path: str
    batch_size: int
    repeat: int
    started_at_utc: str
    client_total_seconds: float
    server_total_seconds: float
    sum_item_seconds: float
    mean_item_seconds: float
    throughput_images_per_second: float
    client_minus_server_seconds: float
    succeeded: int
    failed: int
    response_total: int
    batch_run_id: str | None
    log_total_seconds: float | None
    log_minus_response_seconds: float | None


class BenchmarkError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark OCR batch endpoints by uploading the same image N times "
            "for every N in the requested range."
        )
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8888",
        help="API base URL. Default: %(default)s",
    )
    parser.add_argument(
        "--id-card",
        type=Path,
        default=script_dir / "realidcard.png",
        help="ID-card image path.",
    )
    parser.add_argument(
        "--passport",
        type=Path,
        default=script_dir / "uzpassport.png",
        help="Passport image path.",
    )
    parser.add_argument(
        "--driving-license",
        type=Path,
        default=script_dir / "test_license.jpg",
        help="Driving-licence image path.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=project_root / "logs",
        help=(
            "Artifact log directory. The script cross-checks batch timing.json "
            "when it exists. Default: project-root/logs"
        ),
    )
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=60)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "Requests per endpoint and batch size. Graphs use the median. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Timeout in seconds for one HTTP request. Default: %(default)s",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help=(
            "Do not make one unmeasured request per endpoint before benchmarking. "
            "Use this only when you intentionally want the first measured point "
            "to include lazy model loading."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="Directory for CSV, JSON, and PNG outputs. Default: script directory",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=("id_card", "passport", "driving_license"),
        help="Benchmark only selected endpoints.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.start < 1:
        raise BenchmarkError("--start must be at least 1")
    if args.end < args.start:
        raise BenchmarkError("--end must be greater than or equal to --start")
    if args.step < 1:
        raise BenchmarkError("--step must be at least 1")
    if args.repeats < 1:
        raise BenchmarkError("--repeats must be at least 1")
    if args.timeout <= 0:
        raise BenchmarkError("--timeout must be greater than zero")


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def build_endpoints(args: argparse.Namespace) -> list[EndpointConfig]:
    endpoints = [
        EndpointConfig(
            key="id_card",
            label="ID card MRZ",
            path="/ocr/id-card/mrz/batch",
            image_path=resolve_path(args.id_card),
            log_operation="id_card_mrz_batch",
        ),
        EndpointConfig(
            key="passport",
            label="Passport MRZ",
            path="/ocr/passport/mrz/batch",
            image_path=resolve_path(args.passport),
            log_operation="passport_mrz_batch",
        ),
        EndpointConfig(
            key="driving_license",
            label="Driving licence",
            path="/ocr/driving-license/extract/batch",
            image_path=resolve_path(args.driving_license),
            log_operation="driving_license_batch",
        ),
    ]

    if args.only:
        selected = set(args.only)
        endpoints = [endpoint for endpoint in endpoints if endpoint.key in selected]

    for endpoint in endpoints:
        if not endpoint.image_path.is_file():
            raise BenchmarkError(
                f"{endpoint.label} input does not exist: {endpoint.image_path}"
            )

    return endpoints


def verify_server(
    session: requests.Session,
    base_url: str,
    endpoints: list[EndpointConfig],
    timeout: float,
) -> None:
    openapi_url = f"{base_url.rstrip('/')}/openapi.json"
    try:
        response = session.get(openapi_url, timeout=min(timeout, 30.0))
        response.raise_for_status()
        schema = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise BenchmarkError(
            f"Could not read {openapi_url}. Is Uvicorn running? Error: {exc}"
        ) from exc

    paths = schema.get("paths", {})
    missing = [endpoint.path for endpoint in endpoints if endpoint.path not in paths]
    if missing:
        raise BenchmarkError(
            "The running API does not expose these required batch routes: "
            + ", ".join(missing)
        )


def guess_mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def make_multipart(
    image_path: Path,
    image_bytes: bytes,
    mime_type: str,
    batch_size: int,
) -> list[tuple[str, tuple[str, bytes, str]]]:
    width = max(2, len(str(batch_size)))
    return [
        (
            "files",
            (
                f"{index:0{width}d}_{image_path.name}",
                image_bytes,
                mime_type,
            ),
        )
        for index in range(1, batch_size + 1)
    ]


def read_log_total_seconds(
    logs_dir: Path,
    endpoint: EndpointConfig,
    batch_run_id: str | None,
) -> float | None:
    if not batch_run_id:
        return None

    timing_path = logs_dir / endpoint.log_operation / batch_run_id / "timing.json"
    if not timing_path.is_file():
        return None

    try:
        payload = json.loads(timing_path.read_text(encoding="utf-8"))
        value = payload.get("total_seconds")
        return float(value) if isinstance(value, (int, float)) else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def post_batch(
    session: requests.Session,
    base_url: str,
    endpoint: EndpointConfig,
    image_bytes: bytes,
    mime_type: str,
    batch_size: int,
    repeat: int,
    timeout: float,
    logs_dir: Path,
) -> Measurement:
    url = f"{base_url.rstrip('/')}{endpoint.path}"
    multipart = make_multipart(
        endpoint.image_path,
        image_bytes,
        mime_type,
        batch_size,
    )

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    try:
        response = session.post(
            url,
            headers={"accept": "application/json"},
            files=multipart,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise BenchmarkError(
            f"{endpoint.label}, N={batch_size}: request failed: {exc}"
        ) from exc
    client_seconds = time.perf_counter() - started

    if response.status_code != 200:
        body = response.text[:3000]
        hint = ""
        if response.status_code == 413 and batch_size > 20:
            hint = (
                "\nThe server probably still has BATCH_MAX_FILES=20. "
                "Set BATCH_MAX_FILES=60 or higher in .env and restart Uvicorn."
            )
        raise BenchmarkError(
            f"{endpoint.label}, N={batch_size}: HTTP {response.status_code}: "
            f"{body}{hint}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise BenchmarkError(
            f"{endpoint.label}, N={batch_size}: response was not valid JSON"
        ) from exc

    try:
        server_seconds = float(payload["total_seconds"])
        response_total = int(payload["total"])
        succeeded = int(payload["succeeded"])
        failed = int(payload["failed"])
        items = payload["items"]
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchmarkError(
            f"{endpoint.label}, N={batch_size}: unexpected response schema"
        ) from exc

    if response_total != batch_size:
        raise BenchmarkError(
            f"{endpoint.label}, N={batch_size}: server reported "
            f"{response_total} items instead of {batch_size}"
        )

    item_seconds = [
        float(item["total_seconds"])
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get("total_seconds"), (int, float))
    ]
    sum_item_seconds = sum(item_seconds)
    mean_item_seconds = sum_item_seconds / len(item_seconds) if item_seconds else 0.0
    throughput = batch_size / server_seconds if server_seconds > 0 else float("inf")

    batch_run_id = payload.get("batch_run_id")
    if batch_run_id is not None:
        batch_run_id = str(batch_run_id)

    log_total = read_log_total_seconds(
        logs_dir,
        endpoint,
        batch_run_id,
    )

    return Measurement(
        endpoint=endpoint.key,
        endpoint_label=endpoint.label,
        endpoint_path=endpoint.path,
        batch_size=batch_size,
        repeat=repeat,
        started_at_utc=started_at,
        client_total_seconds=client_seconds,
        server_total_seconds=server_seconds,
        sum_item_seconds=sum_item_seconds,
        mean_item_seconds=mean_item_seconds,
        throughput_images_per_second=throughput,
        client_minus_server_seconds=client_seconds - server_seconds,
        succeeded=succeeded,
        failed=failed,
        response_total=response_total,
        batch_run_id=batch_run_id,
        log_total_seconds=log_total,
        log_minus_response_seconds=(
            log_total - server_seconds if log_total is not None else None
        ),
    )


def linear_regression(
    xs: list[float],
    ys: list[float],
) -> dict[str, float]:
    if len(xs) != len(ys) or not xs:
        raise BenchmarkError("Cannot fit an empty or inconsistent dataset")

    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)

    if denominator == 0:
        slope = 0.0
    else:
        slope = (
            sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
            / denominator
        )

    intercept = y_mean - slope * x_mean
    predicted = [intercept + slope * x for x in xs]
    residual_sum = sum(
        (actual - estimate) ** 2 for actual, estimate in zip(ys, predicted, strict=True)
    )
    total_sum = sum((actual - y_mean) ** 2 for actual in ys)
    r_squared = 1.0 if total_sum == 0 else 1.0 - residual_sum / total_sum

    return {
        "slope_seconds_per_image": slope,
        "intercept_seconds": intercept,
        "r_squared": r_squared,
    }


def aggregate_endpoint(
    records: list[Measurement],
    endpoint_key: str,
) -> dict[str, Any]:
    endpoint_records = [record for record in records if record.endpoint == endpoint_key]
    sizes = sorted({record.batch_size for record in endpoint_records})

    aggregated: list[dict[str, float | int]] = []
    for size in sizes:
        same_size = [record for record in endpoint_records if record.batch_size == size]
        aggregated.append(
            {
                "batch_size": size,
                "server_total_seconds_median": statistics.median(
                    record.server_total_seconds for record in same_size
                ),
                "client_total_seconds_median": statistics.median(
                    record.client_total_seconds for record in same_size
                ),
                "mean_item_seconds_median": statistics.median(
                    record.mean_item_seconds for record in same_size
                ),
                "throughput_images_per_second_median": statistics.median(
                    record.throughput_images_per_second for record in same_size
                ),
            }
        )

    xs = [float(row["batch_size"]) for row in aggregated]
    server_ys = [float(row["server_total_seconds_median"]) for row in aggregated]
    fit = linear_regression(xs, server_ys)

    return {
        "endpoint": endpoint_key,
        "aggregated": aggregated,
        "linear_fit": fit,
    }


def save_csv(path: Path, records: list[Measurement]) -> None:
    if not records:
        return

    rows = [asdict(record) for record in records]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def plot_endpoint(
    output_dir: Path,
    endpoint: EndpointConfig,
    summary: dict[str, Any],
) -> Path:
    aggregated = summary["aggregated"]
    fit = summary["linear_fit"]

    xs = [int(row["batch_size"]) for row in aggregated]
    server = [float(row["server_total_seconds_median"]) for row in aggregated]
    client = [float(row["client_total_seconds_median"]) for row in aggregated]
    fitted = [fit["intercept_seconds"] + fit["slope_seconds_per_image"] * x for x in xs]

    figure = plt.figure(figsize=(11, 7))
    plt.plot(xs, server, marker="o", markersize=3, label="Server total_seconds")
    plt.plot(xs, client, linestyle="--", label="Client wall time")
    plt.plot(xs, fitted, linestyle=":", label="Linear fit of server time")
    plt.xlabel("Images in one batch")
    plt.ylabel("Time (seconds)")
    plt.title(
        f"{endpoint.label} batch complexity\n"
        f"slope={fit['slope_seconds_per_image']:.4f} s/image, "
        f"intercept={fit['intercept_seconds']:.4f} s, "
        f"R²={fit['r_squared']:.6f}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / f"{endpoint.key}_batch_time.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_combined(
    output_dir: Path,
    endpoints: list[EndpointConfig],
    summaries: dict[str, dict[str, Any]],
) -> Path:
    figure = plt.figure(figsize=(12, 8))

    for endpoint in endpoints:
        summary = summaries.get(endpoint.key)
        if not summary:
            continue
        aggregated = summary["aggregated"]
        xs = [int(row["batch_size"]) for row in aggregated]
        ys = [float(row["server_total_seconds_median"]) for row in aggregated]
        fit = summary["linear_fit"]
        label = (
            f"{endpoint.label} "
            f"({fit['slope_seconds_per_image']:.3f} s/image, "
            f"R²={fit['r_squared']:.4f})"
        )
        plt.plot(xs, ys, marker="o", markersize=3, label=label)

    plt.xlabel("Images in one batch")
    plt.ylabel("Server batch time (seconds)")
    plt.title("OCR batch endpoint comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "all_batch_times.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def write_outputs(
    output_dir: Path,
    endpoints: list[EndpointConfig],
    records: list[Measurement],
    run_metadata: dict[str, Any],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    csv_path = output_dir / "benchmark_results.csv"
    json_path = output_dir / "benchmark_results.json"
    summary_path = output_dir / "benchmark_summary.json"

    save_csv(csv_path, records)
    save_json(
        json_path,
        {
            "run": run_metadata,
            "measurements": [asdict(record) for record in records],
        },
    )
    created.extend([csv_path, json_path])

    summaries: dict[str, dict[str, Any]] = {}
    for endpoint in endpoints:
        if any(record.endpoint == endpoint.key for record in records):
            summaries[endpoint.key] = aggregate_endpoint(
                records,
                endpoint.key,
            )

    save_json(
        summary_path,
        {
            "run": run_metadata,
            "endpoints": summaries,
        },
    )
    created.append(summary_path)

    for endpoint in endpoints:
        summary = summaries.get(endpoint.key)
        if summary:
            created.append(plot_endpoint(output_dir, endpoint, summary))

    if summaries:
        created.append(plot_combined(output_dir, endpoints, summaries))

    return created


def print_summary(
    endpoints: list[EndpointConfig],
    records: list[Measurement],
) -> None:
    print("\nLinear-fit summary")
    print("=" * 72)
    for endpoint in endpoints:
        if not any(record.endpoint == endpoint.key for record in records):
            continue
        summary = aggregate_endpoint(records, endpoint.key)
        fit = summary["linear_fit"]
        print(
            f"{endpoint.label:20s} "
            f"slope={fit['slope_seconds_per_image']:.6f} s/image, "
            f"intercept={fit['intercept_seconds']:.6f} s, "
            f"R²={fit['r_squared']:.6f}"
        )


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
        endpoints = build_endpoints(args)
    except BenchmarkError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    base_url = args.base_url.rstrip("/")
    output_dir = resolve_path(args.output_dir)
    logs_dir = resolve_path(args.logs_dir)
    batch_sizes = list(range(args.start, args.end + 1, args.step))
    warmup_enabled = not args.no_warmup

    total_processed_images = sum(batch_sizes) * args.repeats * len(endpoints) + (
        len(endpoints) if warmup_enabled else 0
    )
    total_requests = len(batch_sizes) * args.repeats * len(endpoints) + (
        len(endpoints) if warmup_enabled else 0
    )

    print(f"API: {base_url}")
    print(f"Output directory: {output_dir}")
    print(f"Logs directory: {logs_dir}")
    print(f"Endpoints: {', '.join(endpoint.label for endpoint in endpoints)}")
    print(
        f"Plan: {total_requests} HTTP requests, "
        f"{total_processed_images} total image processings"
    )
    if args.end > 20:
        print(
            "Important: the server must have BATCH_MAX_FILES="
            f"{args.end} or higher and must be restarted after changing .env."
        )
    print(
        "Do not manually set the multipart Content-Type header; requests adds "
        "the required boundary automatically."
    )

    image_payloads: dict[str, tuple[bytes, str]] = {}
    for endpoint in endpoints:
        image_payloads[endpoint.key] = (
            endpoint.image_path.read_bytes(),
            guess_mime_type(endpoint.image_path),
        )

    session = requests.Session()
    records: list[Measurement] = []
    interrupted = False
    failure_message: str | None = None
    run_started = datetime.now(timezone.utc)

    try:
        verify_server(session, base_url, endpoints, args.timeout)

        if warmup_enabled:
            print("\nWarm-up requests (not included in graphs)")
            for endpoint in endpoints:
                image_bytes, mime_type = image_payloads[endpoint.key]
                print(f"  Warming {endpoint.label}...", flush=True)
                post_batch(
                    session=session,
                    base_url=base_url,
                    endpoint=endpoint,
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    batch_size=1,
                    repeat=0,
                    timeout=args.timeout,
                    logs_dir=logs_dir,
                )

        print("\nMeasured requests")
        benchmark_started = time.perf_counter()

        for endpoint in endpoints:
            image_bytes, mime_type = image_payloads[endpoint.key]
            print(f"\n[{endpoint.label}]")
            for batch_size in batch_sizes:
                for repeat in range(1, args.repeats + 1):
                    request_started = time.perf_counter()
                    measurement = post_batch(
                        session=session,
                        base_url=base_url,
                        endpoint=endpoint,
                        image_bytes=image_bytes,
                        mime_type=mime_type,
                        batch_size=batch_size,
                        repeat=repeat,
                        timeout=args.timeout,
                        logs_dir=logs_dir,
                    )
                    records.append(measurement)
                    request_elapsed = time.perf_counter() - request_started

                    status = (
                        "OK"
                        if measurement.failed == 0
                        else f"{measurement.failed} item failures"
                    )
                    log_check = (
                        "no log timing"
                        if measurement.log_total_seconds is None
                        else (f"log Δ={measurement.log_minus_response_seconds:+.6f}s")
                    )
                    print(
                        f"  N={batch_size:02d} repeat={repeat} | "
                        f"server={measurement.server_total_seconds:9.3f}s | "
                        f"client={measurement.client_total_seconds:9.3f}s | "
                        f"{status} | {log_check} | "
                        f"request={request_elapsed:9.3f}s",
                        flush=True,
                    )

        total_benchmark_seconds = time.perf_counter() - benchmark_started
        print(
            f"\nMeasured benchmark completed in {total_benchmark_seconds:.1f} seconds."
        )

    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Saving partial results...", file=sys.stderr)
    except BenchmarkError as exc:
        failure_message = str(exc)
        print(f"\nBenchmark failed: {exc}", file=sys.stderr)
    finally:
        session.close()

        run_completed = datetime.now(timezone.utc)
        run_metadata = {
            "started_at_utc": run_started.isoformat(),
            "completed_at_utc": run_completed.isoformat(),
            "base_url": base_url,
            "batch_start": args.start,
            "batch_end": args.end,
            "batch_step": args.step,
            "repeats": args.repeats,
            "warmup_enabled": warmup_enabled,
            "logs_dir": str(logs_dir),
            "interrupted": interrupted,
            "failure": failure_message,
            "planned_total_requests": total_requests,
            "planned_total_image_processings": total_processed_images,
            "completed_measurements": len(records),
            "input_images": {
                endpoint.key: str(endpoint.image_path) for endpoint in endpoints
            },
        }

        created = write_outputs(
            output_dir,
            endpoints,
            records,
            run_metadata,
        )
        if records:
            print_summary(endpoints, records)
        print("\nSaved files:")
        for path in created:
            if path.exists():
                print(f"  {path}")

    return 1 if failure_message else 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
