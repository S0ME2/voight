import re
from datetime import datetime
from typing import Any

PREFIXES = {
    "surname": r"^\s*1\s*[.\-:,]?\s*", "given_names": r"^\s*2\s*[.\-:,]?\s*",
    "birth_place_and_date": r"^\s*3\s*[.\-:,]?\s*", "issue_date": r"^\s*4\s*a\s*[.\-:,]?\s*",
    "expiry_date": r"^\s*4\s*b\s*[.\-:,]?\s*", "issued_place": r"^\s*4\s*c\s*[.\-:,]?\s*",
    "personal_id": r"^\s*4\s*d\s*[.\-:,]?\s*", "license_number": r"^\s*5\s*[.\-:,]?\s*",
    "address": r"^\s*8\s*[.\-:,]?\s*", "categories": r"^\s*9\s*[.\-:,]?\s*",
}
DATE = re.compile(r"(?<!\d)(\d{1,2})\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{4})(?!\d)")


def _remove_prefix(field: str, value: str) -> str:
    return re.sub(PREFIXES.get(field, ""), "", value.strip(), count=1, flags=re.I).strip()


def _clean_text(value: str) -> str | None:
    value = re.sub(r"\s*,\s*", ", ", " ".join(value.split())).strip()
    return value or None


def _date(value: str) -> str | None:
    match = DATE.search(value)
    if not match:
        return None
    day, month, year = map(int, match.groups())
    try:
        datetime(year, month, day)
    except ValueError:
        return None
    return f"{day:02d}.{month:02d}.{year:04d}"


def _raw_fields(assignments: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    return {
        field: " ".join(token["text"] for token in sorted(tokens, key=lambda token: (token["center_y"], token["x1"])) if token["text"])
        for field, tokens in assignments.items()
    }


def _identifier(value: str) -> str | None:
    value = value.upper()
    return max(re.findall(r"[A-Z]{1,4}\d{4,}", value), key=len, default=re.sub(r"[^A-Z0-9]", "", value)) or None


def parse_fields(assignments: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, str]]:
    raw = _raw_fields(assignments)
    birth = _remove_prefix("birth_place_and_date", raw.get("birth_place_and_date", ""))
    birth_date = DATE.search(birth)
    personal = _remove_prefix("personal_id", raw.get("personal_id", "")).upper().translate(str.maketrans("OQIL", "0011"))
    personal_id = max(re.findall(r"\d{8,}", personal), key=len, default=re.sub(r"\D", "", personal)) or None
    extracted = {
        "surname": _clean_text(_remove_prefix("surname", raw.get("surname", ""))),
        "given_names": _clean_text(_remove_prefix("given_names", raw.get("given_names", ""))),
        "birth_place": _clean_text(birth[:birth_date.start()] if birth_date else birth),
        "birth_date": _date(birth_date.group(0)) if birth_date else None,
        "issue_date": _date(raw.get("issue_date", "")),
        "expiry_date": _date(raw.get("expiry_date", "")),
        "issued_place": _clean_text(_remove_prefix("issued_place", raw.get("issued_place", ""))),
        "personal_id": personal_id,
        "license_number": _identifier(_remove_prefix("license_number", raw.get("license_number", ""))),
        "address": _clean_text(_remove_prefix("address", raw.get("address", ""))),
        "categories": _clean_text(_remove_prefix("categories", raw.get("categories", "")).upper()),
        "serial_number": _identifier(raw.get("serial_number", "")),
    }
    return extracted, raw


def validation_warnings(data: dict[str, Any]) -> list[str]:
    required = ("surname", "given_names", "birth_date", "issue_date", "expiry_date", "personal_id", "license_number")
    warnings = [f"Missing required field: {field}" for field in required if not data.get(field)]
    if data.get("personal_id") and len(data["personal_id"]) != 14:
        warnings.append(f"personal_id has unexpected length: {len(data['personal_id'])}")
    return warnings
