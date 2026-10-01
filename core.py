"""Entry point used by the harness scripts: `triage_submission(submission, model=...)`.

The implementation lives in the `triage` package. This module keeps the harness
contract (function signature, output models, input schema names) stable.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from triage import heuristic
from triage.engine import Extractor, triage
from triage.models import Decision, Evidence, IssueCategory, TriageIssue, TriageOutput

__all__ = [
    "Decision",
    "Evidence",
    "IssueCategory",
    "PatientSubmission",
    "TriageIssue",
    "TriageOutput",
    "triage_output_json_schema",
    "triage_submission",
]

ProcedureRisk = Literal["LOW", "MODERATE", "HIGH"]

# -------------------------
# Input schema (documentation / compatibility)
# -------------------------
# Kept for callers that already hold a PatientSubmission. Triage itself reads the raw
# dict: validating through these models drops unknown fields (e.g. a `value_c`
# temperature) and rejects unexpected enum values, which would lose clinical data.


class PatientName(BaseModel):
    given: str | None = None
    family: str | None = None


class PatientInfo(BaseModel):
    id: str | None = None
    mrn: str | None = None
    name: PatientName | None = None
    dob: str | None = None
    sex: str | None = None


class ProcedureInfo(BaseModel):
    case_id: str | None = None
    procedure_type: str | None = None
    procedure_risk: str | None = None
    procedure_date: str | None = None
    is_elective: bool | None = None
    location: str | None = None


class PatientSubmission(BaseModel):
    """Single submission package shape from the take-home prompt."""

    model_config = {"extra": "allow"}

    patient: PatientInfo | None = None
    procedure: ProcedureInfo | None = None
    vitals: list[dict[str, Any]] = Field(default_factory=list)
    labs: list[dict[str, Any]] = Field(default_factory=list)
    medications: list[dict[str, Any]] = Field(default_factory=list)
    conditions: list[dict[str, Any]] = Field(default_factory=list)
    documents: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None


def triage_output_json_schema() -> dict[str, object]:
    """JSON schema of the triage output contract."""

    return TriageOutput.model_json_schema()


# -------------------------
# Extractor selection
# -------------------------

EXTRACTOR_ENV = "TRIAGE_EXTRACTOR"


def build_extractors(model: str, mode: str | None = None) -> list[tuple[str, Extractor]]:
    """Extractor chain for a run. The heuristic extractor is always the last resort."""

    mode = (mode or os.environ.get(EXTRACTOR_ENV) or "heuristic").lower()
    chain: list[tuple[str, Extractor]] = []
    if mode == "llm":
        from triage.llm import LLMExtractor  # lazy: the heuristic path needs no OpenAI SDK

        chain.append(("llm", LLMExtractor(model=model)))
    elif mode != "heuristic":
        raise ValueError(f"{EXTRACTOR_ENV} must be 'heuristic' or 'llm', got {mode!r}")
    chain.append(("heuristic", heuristic.extract))
    return chain


def triage_submission(
    submission: dict[str, object] | PatientSubmission,
    *,
    model: str,
    extractor: str | None = None,
) -> TriageOutput:
    """Evaluate one pre-op submission package against the Cadence scheduling policy."""

    if isinstance(submission, PatientSubmission):
        submission = submission.model_dump()
    return triage(submission, build_extractors(model, extractor))
