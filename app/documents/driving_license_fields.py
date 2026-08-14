import re
from typing import Any

# Annotation labels are source-layout names; extraction keeps the stable API
# names. Multiple source boxes may intentionally feed one extracted field. The
# OCR text remains raw except for separating the two values on a shared birth line.
FIELD_ALIASES = {
    "name": "given_names",
    "place_of_birth": "birth_place",
    "date_of_birth": "birth_date",
    "date_of_issue": "issue_date",
    "date_of_expiry": "expiry_date",
    "place_of_issue": "issued_place",
    "id_number": "personal_id",
    "id_number_2": "license_number",
    "place_of_living": "address",
    "types": "categories",
}

_DATE = re.compile(r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])[./-](?:0?[1-9]|1[0-2])[./-]\d{4}(?!\d)")


def split_birth_line(value: str) -> tuple[str, str | None]:
    match = _DATE.search(value)
    if not match:
        return value, None
    return value[:match.start()].rstrip(), match.group(0)


def _raw_fields(assignments: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    return {
        field: " ".join(token["text"] for token in sorted(tokens, key=lambda token: (token["center_y"], token["x1"])) if token["text"])
        for field, tokens in assignments.items()
    }


def _canonical_assignments(assignments: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    canonical: dict[str, list[dict[str, Any]]] = {}
    seen: set[object] = set()
    for field, tokens in assignments.items():
        target = canonical.setdefault(FIELD_ALIASES.get(field, field), [])
        for token in tokens:
            marker = token.get("index", id(token))
            if marker in seen:
                continue
            seen.add(marker)
            target.append(token)
    return canonical


def parse_fields(assignments: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, str]]:
    raw = _raw_fields(_canonical_assignments(assignments))
    birth_place, birth_date = split_birth_line(raw.get("birth_place", ""))
    if raw.get("birth_date"):
        birth_date = raw["birth_date"]
    extracted = {
        "surname": raw.get("surname") or None,
        "given_names": raw.get("given_names") or None,
        "patronymic": raw.get("patronymic") or None,
        "birth_place": birth_place or None,
        "birth_date": birth_date or None,
        "issue_date": raw.get("issue_date") or None,
        "expiry_date": raw.get("expiry_date") or None,
        "issued_place": raw.get("issued_place") or None,
        "personal_id": raw.get("personal_id") or None,
        "license_number": raw.get("license_number") or None,
        "address": raw.get("address") or None,
        "categories": raw.get("categories") or None,
        "serial_number": raw.get("serial_number") or None,
    }
    return extracted, raw


def validation_warnings(data: dict[str, Any]) -> list[str]:
    """Reserved for downstream document analysis; recognition emits no warnings."""
    return []
