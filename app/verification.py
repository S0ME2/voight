"""Whole-document OCR verification with conservative, geometry-aware matching."""

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
    line_id: str | int | None = None
    bbox: tuple[float, float, float, float] | None = None
    polygon: tuple[tuple[float, float], ...] | None = None
    reading_order: int | None = None
    side: str | None = None


@dataclass(frozen=True)
class VerificationThresholds:
    candidate_min_score: float = 0.35
    likely_name_score: float = 0.78
    likely_text_score: float = 0.86
    mrz_conflict_score: float = 0.86
    mrz_source_penalty: float = 0.05
    assembly_vertical_gap_factor: float = 3.0
    assembly_left_alignment_factor: float = 0.35
    max_assembly_lines: int = 3
    max_tokens_per_span: int = 6
    assignment_beam_width: int = 64
    # ponytail: cap candidate fan-out at 12; raise after profiling larger documents.
    max_candidates_per_field: int = 12


DEFAULT_THRESHOLDS = VerificationThresholds()


@dataclass(frozen=True)
class _FieldSpec:
    kind: str


@dataclass(frozen=True)
class _Candidate:
    text: str
    resources: tuple[tuple[int, int, int], ...]
    evidence: tuple[dict[str, Any], ...]
    source: str
    source_kind: str
    confidence: float


FIELD_SPECS: dict[str, _FieldSpec] = {
    **{name: _FieldSpec("date") for name in ("birth_date", "issue_date", "expiry_date", "date_of_birth", "date_of_issue", "date_of_expiry")},
    **{name: _FieldSpec("identifier") for name in ("passport_number", "personal_id", "license_number", "serial_number", "card_number", "pinfl")},
    **{name: _FieldSpec("name") for name in ("surname", "given_names", "given_name", "name", "patronymic")},
    **{name: _FieldSpec("text") for name in ("birth_place", "place_of_birth", "issued_place", "place_of_issue", "address", "authority", "nationality", "citizenship", "country_code", "type", "sex", "categories")},
}

_DATE_FIELD_RE = re.compile(r"(?:date|birth|expiry|expire|issue)", re.I)
_ID_FIELD_RE = re.compile(r"(?:number|num|passport|document|card|license|licence|pin|personal|serial|id)", re.I)
_NAME_FIELD_RE = re.compile(r"(?:name|surname|family|given|patronymic)", re.I)
_DATE_SHAPE_RE = re.compile(r"(?<!\w)\d{1,4}[./, -]\d{1,2}[./, -]\d{1,4}(?!\w)|(?<!\w)\d{6,8}(?!\w)")
_TOKEN_RE = re.compile(r"\S+")
_DATE_RE = re.compile(r"(?<!\w)\d{1,4}[./, -]\d{1,2}[./, -]\d{1,4}(?!\w)")
_MRZ_FRAGMENT_RE = re.compile(r"^[A-Z0-9<]+$")
_LABEL_RE = re.compile(r"^\s*(?:(?:\d+[A-Z]|[A-Z]\d*)[.)](?=\s|\d|[A-Z])|\d+[.)](?=\s|[A-Z]))\s*", re.I)


def _text(value: Any) -> str:
    return str(value).strip()


def _kind(field: str, value: str) -> str:
    """Use the supported annotation vocabulary before value-shape fallback."""
    if spec := FIELD_SPECS.get(field.lower()):
        return spec.kind
    if _DATE_FIELD_RE.search(field):
        return "date"
    if _ID_FIELD_RE.search(field):
        return "identifier"
    if _NAME_FIELD_RE.search(field):
        return "name"
    if _DATE_SHAPE_RE.fullmatch(value):
        return "date"
    if sum(char.isdigit() for char in value) >= 2 and any(char.isalpha() for char in value):
        return "identifier"
    return "text"


def _strip_label(value: str, kind: str) -> str:
    stripped = _LABEL_RE.sub("", value, count=1) if kind in {"date", "identifier", "name", "text"} else value
    return stripped or value


def _normalize(value: str, kind: str) -> str:
    value = " ".join(_strip_label(value, kind).upper().split())
    if kind == "date":
        parts = re.split(r"[./, -]+", value)
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
    if kind in {"identifier", "name"}:
        return "".join(char for char in value if char.isalnum())
    return " ".join(re.sub(r"[^\w ]", " ", value, flags=re.UNICODE).split())


def _line_box(line: VerificationLine) -> tuple[float, float, float, float] | None:
    if line.bbox is not None:
        return line.bbox
    if line.polygon:
        xs, ys = zip(*line.polygon)
        return min(xs), min(ys), max(xs), max(ys)
    return None


def _line_source(line: VerificationLine, index: int) -> str:
    return line.source or line.side or "visible_ocr"


def _line_id(line: VerificationLine, index: int) -> str:
    return str(line.line_id if line.line_id is not None else index)


def _evidence(lines: Sequence[VerificationLine], resources: tuple[tuple[int, int, int], ...], source: str) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "line_id": _line_id(lines[index], index),
            "span": [start, end],
            "source": source,
            "side": lines[index].side or (lines[index].source if lines[index].source in {"front", "back"} else None),
            "bbox": list(_line_box(lines[index])) if _line_box(lines[index]) else None,
        }
        for index, start, end in resources
    )


def _candidate(lines: Sequence[VerificationLine], text: str, resources: tuple[tuple[int, int, int], ...], source_kind: str = "visible_ocr") -> _Candidate:
    source = "mrz" if source_kind == "mrz" else _line_source(lines[resources[0][0]], resources[0][0])
    confidence = min(lines[index].confidence for index, _, _ in resources)
    return _Candidate(text.strip(), resources, _evidence(lines, resources, source_kind), source, source_kind, confidence)


def _line_candidates(lines: Sequence[VerificationLine], index: int, thresholds: VerificationThresholds) -> list[_Candidate]:
    line = lines[index]
    value = line.text.strip()
    if not value:
        return []
    found: dict[tuple[str, tuple[tuple[int, int, int], ...]], _Candidate] = {}

    def add(text: str, start: int, end: int) -> None:
        candidate = _candidate(lines, text, ((index, start, end),))
        if candidate.text:
            found[(candidate.text, candidate.resources)] = candidate

    add(value, 0, len(line.text))
    tokens = list(_TOKEN_RE.finditer(line.text))
    for start_token in range(len(tokens)):
        for end_token in range(start_token + 1, min(len(tokens), start_token + thresholds.max_tokens_per_span) + 1):
            add(line.text[tokens[start_token].start():tokens[end_token - 1].end()], tokens[start_token].start(), tokens[end_token - 1].end())
    for match in _DATE_RE.finditer(line.text):
        add(match.group(), match.start(), match.end())
        prefix = line.text[:match.start()].rstrip()
        if prefix:
            add(prefix, 0, match.start())
    return list(found.values())


def _line_order(lines: Sequence[VerificationLine]) -> list[int]:
    return sorted(
        range(len(lines)),
        key=lambda index: (
            _line_box(lines[index])[1] if _line_box(lines[index]) else float("inf"),
            lines[index].reading_order if lines[index].reading_order is not None else index,
        ),
    )


def _same_side(left: VerificationLine, right: VerificationLine) -> bool:
    left_side = left.side or left.source
    right_side = right.side or right.source
    return not left_side or not right_side or left_side == right_side


def _compatible_lines(left: VerificationLine, right: VerificationLine, thresholds: VerificationThresholds) -> bool:
    if not _same_side(left, right):
        return False
    left_box, right_box = _line_box(left), _line_box(right)
    if left_box is None or right_box is None:
        return True
    first, second = (left_box, right_box) if left_box[1] <= right_box[1] else (right_box, left_box)
    height = max(1.0, first[3] - first[1], second[3] - second[1])
    if second[1] - first[3] > thresholds.assembly_vertical_gap_factor * height:
        return False
    horizontal_overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    left_aligned = abs(first[0] - second[0]) <= thresholds.assembly_left_alignment_factor * max(first[2] - first[0], second[2] - second[0], 1.0)
    return horizontal_overlap > 0 or left_aligned


def _assembly_candidates(lines: Sequence[VerificationLine], thresholds: VerificationThresholds) -> list[_Candidate]:
    ordered = _line_order(lines)
    found: dict[tuple[str, tuple[tuple[int, int, int], ...]], _Candidate] = {}
    for position, first in enumerate(ordered):
        group = [first]
        for second in ordered[position + 1:]:
            if (_line_box(lines[group[-1]]) is None or _line_box(lines[second]) is None) and second != group[-1] + 1:
                continue
            if len(group) >= thresholds.max_assembly_lines or not _compatible_lines(lines[group[-1]], lines[second], thresholds):
                if len(group) >= thresholds.max_assembly_lines:
                    break
                continue
            group.append(second)
            resources = tuple((index, 0, len(lines[index].text)) for index in group)
            candidate = _candidate(lines, " ".join(lines[index].text.strip() for index in group), resources)
            found[(candidate.text, candidate.resources)] = candidate
    return list(found.values())


def _mrz_candidates(lines: Sequence[VerificationLine], document_type: str | None, thresholds: VerificationThresholds) -> list[_Candidate]:
    if document_type not in {"passport", "id_card"}:
        return []
    from app.documents.mrz import parse
    eligible = [
        index for index, line in enumerate(lines)
        if (fragment := re.sub(r"[^A-Z0-9<]", "", line.text.upper()))
        and len(fragment) >= 20
        and _MRZ_FRAGMENT_RE.fullmatch(fragment)
    ]
    counts = (2,) if document_type == "passport" else (3,)
    candidates: list[_Candidate] = []
    ordered = sorted(eligible, key=lambda index: (_line_box(lines[index])[1] if _line_box(lines[index]) else index, index))
    for count in counts:
        for start in range(len(ordered) - count + 1):
            indexes = ordered[start:start + count]
            if any((_line_box(lines[left]) is None or _line_box(lines[right]) is None) and right != left + 1 for left, right in zip(indexes, indexes[1:])):
                continue
            if not all(_compatible_lines(lines[left], lines[right], thresholds) for left, right in zip(indexes, indexes[1:])):
                continue
            parsed = parse("\n".join(lines[index].text for index in indexes), document_type)
            if not parsed.fields or not parsed.validations or not all(item.status.value == "passed" for item in parsed.validations):
                continue
            for field_name, value in parsed.fields.items():
                if value:
                    if document_type == "passport":
                        first, second = indexes
                        ranges = {
                            "document_code": (first, 0, 2), "issuing_state": (first, 2, 5),
                            "surname": (first, 5, lines[first].text.find("<<") if "<<" in lines[first].text else 44),
                            "given_names": (first, lines[first].text.find("<<") + 2 if "<<" in lines[first].text else 5, 44),
                            "document_number": (second, 0, 9), "nationality": (second, 10, 13),
                            "date_of_birth": (second, 13, 19), "sex": (second, 20, 21),
                            "date_of_expiry": (second, 21, 27), "optional_data": (second, 28, 42),
                        }
                    else:
                        first, second, third = indexes
                        separator = lines[third].text.find("<<")
                        ranges = {
                            "document_code": (first, 0, 2), "issuing_state": (first, 2, 5),
                            "document_number": (first, 5, 14), "optional_data": (first, 15, 30), "pinfl": (first, 15, 30),
                            "date_of_birth": (second, 0, 6), "sex": (second, 7, 8), "date_of_expiry": (second, 8, 14),
                            "nationality": (second, 15, 18), "surname": (third, 0, separator if separator >= 0 else 30),
                            "given_names": (third, separator + 2 if separator >= 0 else 0, 30),
                        }
                    resource = ranges.get(field_name)
                    resources = (resource,) if resource else tuple((index, 0, len(lines[index].text)) for index in indexes)
                    values = [str(value)]
                    if field_name in {"date_of_birth", "date_of_expiry"} and len(str(value)) == 6 and str(value).isdigit():
                        year, month, day = int(str(value)[:2]), int(str(value)[2:4]), int(str(value)[4:])
                        year += 1900 if year >= 50 else 2000
                        values.append(f"{day:02d}.{month:02d}.{year:04d}")
                    candidates.extend(_candidate(lines, candidate_value, resources, "mrz") for candidate_value in values)
    return candidates


def _candidates(
    lines: Sequence[VerificationLine],
    document_type: str | None,
    thresholds: VerificationThresholds,
    instrumentation: dict[str, Any] | None = None,
) -> list[_Candidate]:
    started = time.perf_counter()
    values = [candidate for index in range(len(lines)) for candidate in _line_candidates(lines, index, thresholds)]
    if instrumentation is not None:
        instrumentation["token_span_construction_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    values.extend(_assembly_candidates(lines, thresholds))
    if instrumentation is not None:
        instrumentation["geometry_assembly_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    values.extend(_mrz_candidates(lines, document_type, thresholds))
    if instrumentation is not None:
        instrumentation["mrz_validation_seconds"] = time.perf_counter() - started
    seen = set()
    return [candidate for candidate in values if not (key := (candidate.text, candidate.resources, candidate.source_kind)) in seen and not seen.add(key)]


def _shape_compatible(expected: str, detected: str, kind: str) -> bool:
    normalized = _normalize(detected, kind)
    target = _normalize(expected, kind)
    if not normalized or not target:
        return False
    if kind == "date":
        return bool(_DATE_SHAPE_RE.search(detected) or (normalized.isdigit() and len(normalized) in {6, 8}))
    if kind == "identifier":
        if target.isdigit():
            return normalized.isdigit()
        return normalized.isalnum() and any(char.isalpha() for char in normalized) and any(char.isdigit() for char in normalized)
    if kind == "name" and normalized.isdigit() and not target.isdigit():
        return False
    return True


def _score(expected: str, detected: str, kind: str) -> float:
    if not _shape_compatible(expected, detected, kind):
        return 0.0
    left, right = _normalize(expected, kind), _normalize(detected, kind)
    return SequenceMatcher(None, left, right).ratio() if left and right else 0.0


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


def _overlaps(candidate: _Candidate, used: tuple[tuple[int, int, int], ...]) -> bool:
    for left_index, left_start, left_end in candidate.resources:
        for right_index, right_start, right_end in used:
            if left_index == right_index and max(left_start, right_start) < min(left_end, right_end):
                return True
    return False


def verify_fields(
    lines: Sequence[VerificationLine],
    expected_fields: Mapping[str, Any],
    thresholds: VerificationThresholds = DEFAULT_THRESHOLDS,
    *,
    document_type: str | None = None,
    instrumentation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assign non-overlapping OCR spans globally, preserving source evidence."""
    started = time.perf_counter()
    candidates = _candidates(lines, document_type, thresholds, instrumentation)
    if instrumentation is not None:
        instrumentation["candidate_generation_seconds"] = time.perf_counter() - started
        instrumentation["candidate_count"] = len(candidates)
    started = time.perf_counter()
    fields = [(name, _text(value), _kind(name, _text(value))) for name, value in expected_fields.items()]
    if instrumentation is not None:
        instrumentation["normalization_seconds"] = time.perf_counter() - started
        instrumentation["field_count"] = len(fields)
    options: list[list[tuple[int, float]]] = []
    evidence: list[dict[str, Any]] = []
    started = time.perf_counter()
    score_comparisons = 0
    for field_name, expected, kind in fields:
        scored = [(index, _score(expected, candidate.text, kind)) for index, candidate in enumerate(candidates)]
        score_comparisons += len(scored)
        visible_best = max((score for index, score in scored if candidates[index].source_kind == "visible_ocr"), default=0.0)
        adjusted = [
            (index, score - thresholds.mrz_source_penalty if candidates[index].source_kind == "mrz" and visible_best >= thresholds.mrz_conflict_score else score)
            for index, score in scored
        ]
        adjusted.sort(key=lambda item: (item[1], scored[item[0]][1], candidates[item[0]].confidence), reverse=True)
        if instrumentation is not None:
            evidence.append({
                "field": field_name, "expected": expected, "kind": kind,
                "candidates": [
                    {"text": candidates[index].text, "resources": candidates[index].resources, "evidence": candidates[index].evidence, "source": candidates[index].source, "source_kind": candidates[index].source_kind, "score": round(raw_score, 4), "eligible": raw_score >= thresholds.candidate_min_score}
                    for index, raw_score in sorted(scored, key=lambda item: item[1], reverse=True)
                ],
            })
        options.append([(index, raw_score) for index, raw_score in adjusted if raw_score >= thresholds.candidate_min_score][:thresholds.max_candidates_per_field])
    if instrumentation is not None:
        instrumentation["similarity_scoring_seconds"] = time.perf_counter() - started
        instrumentation["score_comparison_count"] = score_comparisons
        instrumentation["option_counts"] = [len(value) for value in options]
        instrumentation["candidate_evidence"] = evidence

    started = time.perf_counter()
    assignment_states = 0
    assignment_transitions = 0
    beam: list[tuple[float, tuple[tuple[int, int, int], ...], tuple[int | None, ...]]] = [(0.0, (), ())]
    for field_options in options:
        next_beam = [(score, used, (*assignment, None)) for score, used, assignment in beam]
        for score, used, assignment in beam:
            for candidate_index, candidate_score in field_options:
                assignment_transitions += 1
                candidate = candidates[candidate_index]
                if _overlaps(candidate, used):
                    continue
                next_used = tuple(sorted((*used, *candidate.resources)))
                next_beam.append((score + candidate_score, next_used, (*assignment, candidate_index)))
        next_beam.sort(key=lambda item: item[0], reverse=True)
        beam = next_beam[:thresholds.assignment_beam_width]
        assignment_states += len(beam)
    assignment = max(beam, default=(0.0, (), ()), key=lambda item: item[0])[2]
    if instrumentation is not None:
        instrumentation["assignment_seconds"] = time.perf_counter() - started
        instrumentation["assignment_states"] = assignment_states
        instrumentation["assignment_transitions"] = assignment_transitions
    started = time.perf_counter()
    result = {}
    strict_elapsed = 0.0
    for index, (name, expected, kind) in enumerate(fields):
        field_started = time.perf_counter() if kind in {"date", "identifier"} else None
        candidate = candidates[assignment[index]] if index < len(assignment) and assignment[index] is not None else None
        score = _score(expected, candidate.text, kind) if candidate else 0.0
        result[name] = {
            "expected": expected,
            "detected": candidate.text if candidate else None,
            "score": round(score, 4),
            "score_source": "normalized_sequence_similarity",
            "status": _status(expected, candidate.text if candidate else None, score, kind, thresholds),
            "source": candidate.source if candidate else None,
            "evidence": list(candidate.evidence) if candidate else [],
        }
        if field_started is not None:
            strict_elapsed += time.perf_counter() - field_started
    if instrumentation is not None:
        instrumentation["strict_field_validation_seconds"] = strict_elapsed
    if instrumentation is not None:
        instrumentation["status_classification_seconds"] = time.perf_counter() - started
    return {
        "fields": result,
        "summary": {status: sum(value["status"] == status for value in result.values()) for status in ("match", "likely_match", "mismatch", "not_found")},
    }
