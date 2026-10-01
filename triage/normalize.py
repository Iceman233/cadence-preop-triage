"""Lenient parsing of a raw submission into indexed records.

The raw dict stays the source of truth: nothing here raises on bad or unexpected values,
and every record keeps its original list index so evidence can cite `labs[3]` exactly.
Values that cannot be interpreted become None and surface later as UNKNOWN.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from . import policy

BLOOD_PRESSURE = "BLOOD_PRESSURE"
TEMPERATURE = "TEMPERATURE"


@dataclass(frozen=True)
class Timestamp:
    """A point in time at day resolution, with minutes when the source has a time."""

    day: date
    minutes: int | None = None

    def __str__(self) -> str:
        if self.minutes is None:
            return self.day.isoformat()
        return f"{self.day.isoformat()} {self.minutes // 60:02d}:{self.minutes % 60:02d}"


@dataclass(frozen=True)
class Procedure:
    type: str | None
    risk: str | None  # LOW / MODERATE / HIGH, or None if missing or unrecognized
    risk_raw: Any
    date: date | None
    date_raw: Any


@dataclass(frozen=True)
class Vital:
    index: int
    kind: str | None
    systolic: float | None
    diastolic: float | None
    temp_f: float | None
    temp_raw: str | None  # the value as written, e.g. "38.6 C"
    at: Timestamp | None


@dataclass(frozen=True)
class Lab:
    index: int
    test: str | None  # canonical policy test name (CBC / CMP) or None
    code: str | None
    display: str | None
    status: str | None
    effective: date | None
    effective_raw: Any


@dataclass(frozen=True)
class Medication:
    index: int
    name: str
    active: bool | None
    active_raw: Any
    generic: str | None
    drug_class: str | None


@dataclass(frozen=True)
class Document:
    index: int
    type: str
    date: date | None
    date_raw: Any
    text: str


@dataclass(frozen=True)
class Case:
    procedure: Procedure
    vitals: list[Vital] = field(default_factory=list)
    labs: list[Lab] = field(default_factory=list)
    medications: list[Medication] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)
    review_at: Timestamp | None = None


# -------------------------
# Scalar coercion
# -------------------------

_US_DATE = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*$")


def parse_timestamp(value: Any) -> Timestamp | None:
    """Parse ISO dates/datetimes (normalized to UTC) and MM/DD/YYYY."""

    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    us = _US_DATE.match(text)
    if us:
        try:
            return Timestamp(date(int(us.group(3)), int(us.group(1)), int(us.group(2))))
        except ValueError:
            return None
    if len(text) == 10:
        try:
            return Timestamp(date.fromisoformat(text))
        except ValueError:
            return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
    return Timestamp(parsed.date(), parsed.hour * 60 + parsed.minute)


def parse_date(value: Any) -> date | None:
    ts = parse_timestamp(value)
    return ts.day if ts else None


def to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "active", "y"}:
            return True
        if lowered in {"false", "no", "inactive", "discontinued", "stopped", "n"}:
            return False
    return None


def _str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


# -------------------------
# Domain parsing
# -------------------------


def fahrenheit(value: float, unit: str | None) -> float:
    """Convert to Fahrenheit; a missing unit is inferred from the magnitude."""

    unit = (unit or "").strip().upper().lstrip("°")
    if unit.startswith("C") or (not unit and value < policy.CELSIUS_INFERENCE_CEILING):
        return round(value * 9 / 5 + 32, 1)
    return value


def lab_test(code: str | None, display: str | None) -> str | None:
    normalized = (code or "").strip().upper()
    for test, (codes, _) in policy.LAB_ALIASES.items():
        if normalized in codes:
            return test
    # Fall back to the display text only when the code is not a known different test.
    for test, (_, display_re) in policy.LAB_ALIASES.items():
        if display and display_re.search(display):
            return test
    return None


def _vital(index: int, raw: dict[str, Any]) -> Vital:
    kind_raw = (_str(raw.get("type")) or "").lower()
    systolic = to_float(raw.get("systolic"))
    diastolic = to_float(raw.get("diastolic"))
    temp_f = temp_raw = None
    for key, unit in (("value_f", "F"), ("value_c", "C"), ("value", _str(raw.get("unit")))):
        value = to_float(raw.get(key))
        if value is not None:
            temp_f = fahrenheit(value, unit)
            temp_raw = f"{key}={raw.get(key)}" + (f" {unit}" if key == "value" and unit else "")
            break

    if "pressure" in kind_raw or kind_raw in {"bp", "nibp"}:
        kind = BLOOD_PRESSURE
    elif "temp" in kind_raw:
        kind = TEMPERATURE
    elif systolic is not None or diastolic is not None:
        kind = BLOOD_PRESSURE
    elif temp_f is not None:
        kind = TEMPERATURE
    else:
        kind = None
    return Vital(
        index=index,
        kind=kind,
        systolic=systolic,
        diastolic=diastolic,
        temp_f=temp_f,
        temp_raw=temp_raw,
        at=parse_timestamp(raw.get("date") or raw.get("effective_at")),
    )


def parse_case(submission: dict[str, Any]) -> Case:
    submission = _dict(submission)
    proc_raw = _dict(submission.get("procedure"))
    risk_raw = proc_raw.get("procedure_risk")
    risk = _str(risk_raw)
    risk = risk.upper() if risk and risk.upper() in policy.REQUIRED_TESTS else None

    procedure = Procedure(
        type=_str(proc_raw.get("procedure_type")),
        risk=risk,
        risk_raw=risk_raw,
        date=parse_date(proc_raw.get("procedure_date")),
        date_raw=proc_raw.get("procedure_date"),
    )

    vitals = [_vital(i, v) for i, v in enumerate(_list(submission.get("vitals"))) if isinstance(v, dict)]

    labs = []
    for i, raw in enumerate(_list(submission.get("labs"))):
        if not isinstance(raw, dict):
            continue
        code, display = _str(raw.get("code")), _str(raw.get("display"))
        labs.append(
            Lab(
                index=i,
                test=lab_test(code, display),
                code=code,
                display=display,
                status=(_str(raw.get("status")) or "").lower() or None,
                effective=parse_date(raw.get("effective_at") or raw.get("date")),
                effective_raw=raw.get("effective_at") or raw.get("date"),
            )
        )

    medications = []
    for i, raw in enumerate(_list(submission.get("medications"))):
        if not isinstance(raw, dict):
            continue
        name = _str(raw.get("name")) or ""
        generic, drug_class = policy.classify_drug(name)
        medications.append(
            Medication(
                index=i,
                name=name,
                active=to_bool(raw.get("active")),
                active_raw=raw.get("active"),
                generic=generic,
                drug_class=drug_class,
            )
        )

    documents = [
        Document(
            index=i,
            type=_str(raw.get("type")) or "",
            date=parse_date(raw.get("date")),
            date_raw=raw.get("date"),
            text=raw.get("text") if isinstance(raw.get("text"), str) else "",
        )
        for i, raw in enumerate(_list(submission.get("documents")))
        if isinstance(raw, dict)
    ]

    metadata = _dict(submission.get("metadata"))
    return Case(
        procedure=procedure,
        vitals=vitals,
        labs=labs,
        medications=medications,
        documents=documents,
        review_at=parse_timestamp(metadata.get("submission_received_at")),
    )
