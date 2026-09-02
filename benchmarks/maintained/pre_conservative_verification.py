"""Whole-document OCR and conservative expected-value verification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
import re
import time
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class VerificationLine:
    text: str
    confidence: float
    source: str | None = None


@dataclass(frozen=True)
class VerificationThresholds:
    candidate_min_score: float = 0.35
    likely_name_score: float = 0.78
    likely_text_score: float = 0.86
    # ponytail: cap candidate fan-out at 12; raise after profiling larger documents.
    max_candidates_per_field: int = 12


DEFAULT_THRESHOLDS = VerificationThresholds()


@dataclass(frozen=True)
class _Candidate:
    text: str
    indexes: tuple[int, ...]
    source: str | None


_DATE_RE = re.compile(r"^\d{1,4}[./ -]\d{1,2}[./ -]\d{1,4}$")
_DATE_FIELD_RE = re.compile(r"(?:date|birth|expiry|expire|issue)", re.I)
_ID_FIELD_RE = re.compile(r"(?:number|num|passport|document|card|license|licence|pin|personal|serial|id)", re.I)
_NAME_FIELD_RE = re.compile(r"(?:name|surname|family|given|patronymic)", re.I)


def _text(value: Any) -> str:
    return str(value).strip()


def _kind(field: str, value: str) -> str:
    if _DATE_RE.fullmatch(value) or (_DATE_FIELD_RE.search(field) and value.isdigit() and len(value) in {6, 8}):
        return "date"
    if _ID_FIELD_RE.search(field) or (sum(char.isdigit() for char in value) >= 2 and any(char.isalpha() for char in value)):
        return "identifier"
    if _NAME_FIELD_RE.search(field):
        return "name"
    return "text"


def _normalize(value: str, kind: str) -> str:
    value = " ".join(value.upper().split())
    if kind == "date":
        parts = re.split(r"[./ -]+", value)
        if len(parts) == 1 and len(parts[0]) in {6, 8} and parts[0].isdigit():
            parts = [parts[0][:2], parts[0][2:4], parts[0][4:]]
        if len(parts) == 3:
            try:
                numbers = tuple(map(int, parts))
                if len(parts[0]) == 4:
                    return date(numbers[0], numbers[1], numbers[2]).isoformat()
                year = numbers[2] + (2000 if numbers[2] < 100 else 0)
                return date(year, numbers[1], numbers[0]).isoformat()
            except ValueError:
                pass
        return re.sub(r"\D", "", value)
    if kind == "identifier":
        return "".join(char for char in value if char.isalnum())
    if kind == "name":
        return "".join(char for char in value if char.isalnum())
    return " ".join(re.sub(r"[^\w ]", " ", value, flags=re.UNICODE).split())


def _candidates(lines: Sequence[VerificationLine]) -> list[_Candidate]:
    values = [line.text.strip() for line in lines]
    candidates = [_Candidate(value, (index,), lines[index].source) for index, value in enumerate(values) if value]
    # A two-line span covers labels/values split by the detector without making
    # the matcher infer document geometry or field positions.
    candidates.extend(
        _Candidate(f"{values[index]} {values[index + 1]}", (index, index + 1), lines[index].source)
        for index in range(len(values) - 1)
        if values[index] and values[index + 1] and lines[index].source == lines[index + 1].source
    )
    return candidates


def _score(expected: str, detected: str, kind: str) -> float:
    left, right = _normalize(expected, kind), _normalize(detected, kind)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _status(expected: str, detected: str | None, score: float, kind: str, thresholds: VerificationThresholds) -> str:
    if detected is None:
        return "not_found"
    if _normalize(expected, kind) == _normalize(detected, kind):
        return "match"
    if kind in {"date", "identifier"}:
        return "mismatch"
    threshold = thresholds.likely_name_score if kind == "name" else thresholds.likely_text_score
    if len(_normalize(expected, kind)) < 4:
        return "mismatch"
    return "likely_match" if score >= threshold else "mismatch"


def verify_fields(
    lines: Sequence[VerificationLine],
    expected_fields: Mapping[str, Any],
    thresholds: VerificationThresholds = DEFAULT_THRESHOLDS,
    *,
    instrumentation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assign non-overlapping OCR line spans globally, then classify evidence."""
    started = time.perf_counter()
    candidates = _candidates(lines)
    if instrumentation is not None:
        instrumentation["candidate_generation_seconds"] = time.perf_counter() - started
        instrumentation["candidate_count"] = len(candidates)
    started = time.perf_counter()
    fields = [(name, _text(value), _kind(name, _text(value))) for name, value in expected_fields.items()]
    if instrumentation is not None:
        instrumentation["normalization_seconds"] = time.perf_counter() - started
        instrumentation["field_count"] = len(fields)
    options = []
    started = time.perf_counter()
    score_comparisons = 0
    for field_name, expected, kind in fields:
        score_comparisons += len(candidates)
        scored = sorted(
            ((candidate, _score(expected, candidate.text, kind)) for candidate in candidates),
            key=lambda item: item[1],
            reverse=True,
        )
        if instrumentation is not None:
            instrumentation.setdefault("candidate_evidence", []).append({
                "field": field_name,
                "expected": expected,
                "kind": kind,
                "candidates": [
                    {"text": candidate.text, "indexes": candidate.indexes, "source": candidate.source, "score": round(score, 4), "eligible": score >= thresholds.candidate_min_score}
                    for candidate, score in scored
                ],
            })
        options.append(
            [(candidate, score) for candidate, score in scored if score >= thresholds.candidate_min_score]
            [: thresholds.max_candidates_per_field]
        )
    if instrumentation is not None:
        instrumentation["similarity_scoring_seconds"] = time.perf_counter() - started
        instrumentation["score_comparison_count"] = score_comparisons
        instrumentation["option_counts"] = [len(value) for value in options]

    from functools import lru_cache
    assignment_states = 0
    assignment_transitions = 0

    @lru_cache(maxsize=None)
    def assign(field_index: int, used_mask: int) -> tuple[float, tuple[int | None, ...]]:
        nonlocal assignment_states, assignment_transitions
        assignment_states += 1
        if field_index == len(fields):
            return 0.0, ()
        best_score, best_assignment = assign(field_index + 1, used_mask)
        for candidate, score in options[field_index]:
            assignment_transitions += 1
            candidate_index = candidates.index(candidate)
            mask = sum(1 << index for index in candidate.indexes)
            if used_mask & mask:
                continue
            total, assignment = assign(field_index + 1, used_mask | mask)
            if score + total > best_score:
                best_score, best_assignment = score + total, (candidate_index, *assignment)
        return best_score, best_assignment

    started = time.perf_counter()
    _, assignment = assign(0, 0)
    if instrumentation is not None:
        instrumentation["assignment_seconds"] = time.perf_counter() - started
        instrumentation["assignment_states"] = assignment_states
        instrumentation["assignment_transitions"] = assignment_transitions
    started = time.perf_counter()
    result = {}
    for index, (name, expected, kind) in enumerate(fields):
        candidate = candidates[assignment[index]] if index < len(assignment) and assignment[index] is not None else None
        score = _score(expected, candidate.text, kind) if candidate else 0.0
        result[name] = {
            "expected": expected,
            "detected": candidate.text if candidate else None,
            "score": round(score, 4),
            "status": _status(expected, candidate.text if candidate else None, score, kind, thresholds),
            "source": candidate.source if candidate else None,
        }
    if instrumentation is not None:
        instrumentation["status_classification_seconds"] = time.perf_counter() - started
    return {
        "fields": result,
        "summary": {status: sum(value["status"] == status for value in result.values()) for status in ("match", "likely_match", "mismatch", "not_found")},
    }
