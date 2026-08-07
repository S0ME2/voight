from typing import Any

# Annotation labels are source-layout names; extraction keeps the stable API
# names. Multiple source boxes may intentionally feed one extracted field. The
# values themselves are never cleaned, parsed, or otherwise interpreted here.
FIELD_ALIASES = {
    "name": "given_names",
    "place_of_birth": "birth_place_and_date",
    "date_of_birth": "birth_place_and_date",
    "date_of_issue": "issue_date",
    "date_of_expiry": "expiry_date",
    "place_of_issue": "issued_place",
    "id_number": "personal_id",
    "id_number_2": "license_number",
    "place_of_living": "address",
    "types": "categories",
}


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
    extracted = {
        "surname": raw.get("surname") or None,
        "given_names": raw.get("given_names") or None,
        "birth_place": raw.get("birth_place_and_date") or None,
        "birth_date": raw.get("date_of_birth") or None,
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
