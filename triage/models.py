"""Output contract and the internal finding type rules produce."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel

from .normalize import Document
from .schema import DocumentFacts

Decision = Literal["READY", "NEEDS_FOLLOW_UP", "NOT_CLEARED"]
IssueCategory = Literal[
    "REQUIRED_DOCUMENTATION",
    "REQUIRED_TESTING",
    "ANTICOAGULATION_MANAGEMENT",
    "ACUTE_SAFETY_EXCLUSION",
    "MISSING_REQUIRED_DATA",
]

# Stable output order for issues.
CATEGORY_ORDER: tuple[str, ...] = (
    "ACUTE_SAFETY_EXCLUSION",
    "MISSING_REQUIRED_DATA",
    "REQUIRED_DOCUMENTATION",
    "REQUIRED_TESTING",
    "ANTICOAGULATION_MANAGEMENT",
)


class Evidence(BaseModel):
    source: str
    details: str


class TriageIssue(BaseModel):
    category: IssueCategory
    description: str
    evidence: Evidence


class TriageOutput(BaseModel):
    decision: Decision
    issues: list[TriageIssue]
    explanation: str


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    # Input needed to evaluate the check is absent or could not be verified.
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Finding:
    """One check's outcome. FAIL maps to `category`; UNKNOWN maps to MISSING_REQUIRED_DATA."""

    rule: str
    status: Status
    category: str
    description: str
    source: str
    details: str
    # Set when UNKNOWN only because a field already reported as missing (e.g. the
    # procedure date) blocks evaluation; such findings fold into that field's issue.
    blocked_by: str | None = None


@dataclass(frozen=True)
class ExtractedDocument:
    doc: Document
    facts: DocumentFacts
    extractor: str  # "llm" | "heuristic" | "none" (extraction failed)

    @property
    def source(self) -> str:
        return f"documents[{self.doc.index}]"

    def label(self) -> str:
        return f"{self.source} ({self.doc.type or 'untyped'}, {self.doc.date_raw})"
