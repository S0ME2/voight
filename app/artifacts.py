import json
import os
import time
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import ArtifactSettings

COUNTER_PREFIX = re.compile(r"^(\d+)_")
UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(filename: str | None) -> str:
    stem = Path(filename or "file").stem.strip()
    safe = UNSAFE_FILENAME_CHARS.sub("-", stem).strip("._-")
    return safe or "file"


def next_counter(parent: Path) -> int:
    if not parent.exists():
        return 1
    directories = [path for path in parent.iterdir() if path.is_dir()]
    numbered = [
        int(match.group(1))
        for path in directories
        if (match := COUNTER_PREFIX.match(path.name))
    ]
    return max(len(directories), max(numbered, default=0)) + 1


def new_run_id(root: Path, operation: str, filename: str | None, create: bool) -> str:
    parent = root / operation
    timestamp = datetime.now().strftime("%H-%M-%S-%f_%d-%m-%Y")
    if not create:
        return f"{next_counter(parent)}_{safe_filename(filename)}_{timestamp}"

    parent.mkdir(parents=True, exist_ok=True)
    while True:
        run_id = f"{next_counter(parent)}_{safe_filename(filename)}_{timestamp}"
        try:
            (parent / run_id).mkdir(exist_ok=False)
        except FileExistsError:
            timestamp = datetime.now().strftime("%H-%M-%S-%f_%d-%m-%Y")
            continue
        return run_id


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return str(value)


@dataclass
class ArtifactWriter:
    root: Path
    operation: str
    run_id: str
    enabled: bool
    profile: dict[str, int] | None = None

    def _record(self, name: str, started_ns: int) -> None:
        if self.profile is not None:
            self.profile[f"artifact.{name}_ns"] = self.profile.get(f"artifact.{name}_ns", 0) + time.perf_counter_ns() - started_ns
            self.profile[f"artifact.{name}_count"] = self.profile.get(f"artifact.{name}_count", 0) + 1

    @property
    def directory(self) -> Path:
        if not self.operation:
            return self.root / self.run_id
        return self.root / self.operation / self.run_id

    def path(self, name: str) -> Path:
        return self.directory / name

    def save_bytes(self, name: str, data: bytes) -> None:
        if self.enabled:
            started = time.perf_counter_ns()
            self.path(name).write_bytes(data)
            self._record("bytes_write", started)

    def save_json(self, name: str, data: Any) -> None:
        if self.enabled:
            started = time.perf_counter_ns()
            encoded = json.dumps(data, ensure_ascii=False, indent=2, default=json_default)
            self._record("json_encode", started)
            started = time.perf_counter_ns()
            self.path(name).write_text(encoded, encoding="utf-8")
            self._record("json_write", started)

    def save_text(self, name: str, text: str) -> None:
        if self.enabled:
            started = time.perf_counter_ns()
            self.path(name).write_text(text, encoding="utf-8")
            self._record("text_write", started)

    def save_image(self, name: str, image: np.ndarray) -> None:
        if self.enabled:
            started = time.perf_counter_ns()
            cv2.imwrite(str(self.path(name)), image)
            self._record("image_write", started)

    def save_model_image(self, name: str, result: Any) -> None:
        if self.enabled:
            started = time.perf_counter_ns()
            result.save_to_img(str(self.path(name)))
            self._record("model_image_write", started)

    def save_model_json(self, name: str, result: Any) -> None:
        if self.enabled:
            started = time.perf_counter_ns()
            result.save_to_json(str(self.path(name)))
            self._record("model_json_write", started)


def create_artifact_run(
    settings: ArtifactSettings,
    operation: str,
    filename: str | None,
    profile: dict[str, int] | None = None,
) -> ArtifactWriter:
    return ArtifactWriter(
        settings.directory,
        operation,
        new_run_id(settings.directory, operation, filename, settings.enabled),
        settings.enabled,
        profile if profile is not None else ({} if os.getenv("VOIGHT_BENCHMARK_PROFILE") else None),
    )


def create_batch_artifact_run(
    settings: ArtifactSettings,
    operation: str,
    profile: dict[str, int] | None = None,
) -> ArtifactWriter:
    """Create one parent directory for an entire batch request."""

    return create_artifact_run(settings, f"{operation}_batch", "batch", profile)


def create_child_artifact_run(
    parent: ArtifactWriter,
    index: int,
    filename: str | None,
) -> ArtifactWriter:
    """Create a deterministic, order-preserving image directory inside a batch."""

    run_id = f"{index + 1:03d}_{safe_filename(filename)}"
    if parent.enabled:
        (parent.directory / run_id).mkdir(parents=True, exist_ok=True)
    return ArtifactWriter(parent.directory, "", run_id, parent.enabled, parent.profile)


def save_input(
    writer: ArtifactWriter,
    data: bytes,
    filename: str | None,
    content_type: str | None,
    extension: str | None,
    image: np.ndarray | None = None,
    *,
    source_filename: str | None = None,
    archive_path: str | None = None,
) -> None:
    writer.save_bytes(f"00_input.{extension or 'bin'}", data)
    metadata: dict[str, Any] = {
        "run_id": writer.run_id,
        "original_filename": filename,
        "declared_content_type": content_type,
        "detected_extension": extension,
        "size_bytes": len(data),
    }
    if source_filename is not None:
        metadata["source_filename"] = source_filename
    if archive_path is not None:
        metadata["archive_path"] = archive_path
    if image is not None:
        metadata.update(
            image_width=int(image.shape[1]),
            image_height=int(image.shape[0]),
            image_channels=int(image.shape[2]) if image.ndim == 3 else 1,
        )
    writer.save_json("00_input_metadata.json", metadata)
