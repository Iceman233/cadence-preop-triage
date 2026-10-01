from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from core import triage_submission

FIXTURES = Path(__file__).parent / "fixtures"

SIGNED_CONSENT = (
    "INFORMED CONSENT FOR SURGICAL PROCEDURE\n"
    "Procedure: carpal tunnel release\n"
    "Patient signature on file - signed in clinic 03/05/2026 09:00 (scanned copy attached).\n"
    "Physician attestation: risks explained. Electronically signed by Amy Doe, MD on 03/05/2026."
)

_READY: dict[str, Any] = {
    "procedure": {
        "procedure_type": "Elective carpal tunnel release",
        "procedure_risk": "LOW",
        "procedure_date": "2026-03-20",
    },
    "vitals": [
        {"type": "blood_pressure", "systolic": 120, "diastolic": 78, "date": "2026-03-10T10:00:00Z"},
        {"type": "temperature", "value_f": 98.4, "date": "2026-03-10T10:05:00Z"},
    ],
    "labs": [{"code": "CBC", "display": "Complete Blood Count", "effective_at": "2026-03-01T08:00:00Z", "status": "final"}],
    "medications": [],
    "conditions": [],
    "documents": [
        {
            "type": "History and Physical",
            "date": "2026-03-01",
            "text": "PREOPERATIVE HISTORY AND PHYSICAL\nReferred for pre-operative evaluation prior to carpal tunnel release.",
        },
        {"type": "Surgical Consent", "date": "2026-03-05", "text": SIGNED_CONSENT},
    ],
    "metadata": {"submission_received_at": "2026-03-20T14:00:00Z"},
}


@pytest.fixture
def ready() -> dict[str, Any]:
    """A submission that satisfies every rule; tests mutate one thing at a time."""

    return copy.deepcopy(_READY)


@pytest.fixture
def run() -> Callable[[dict[str, Any]], Any]:
    return lambda submission: triage_submission(submission, model="unused", extractor="heuristic")


def categories(output) -> set[str]:
    return {issue.category for issue in output.issues}


def add_doc(submission: dict[str, Any], type_: str, date: str, text: str) -> None:
    submission["documents"].append({"type": type_, "date": date, "text": text})


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text())
