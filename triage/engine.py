"""Orchestration: parse -> extract -> rules + tripwires -> decision -> rendered output.

Everything after extraction is deterministic, so the same facts always produce
byte-identical output.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Sequence

from . import heuristic, rules, tripwires
from .models import (
    CATEGORY_ORDER,
    Evidence,
    ExtractedDocument,
    Finding,
    Status,
    TriageIssue,
    TriageOutput,
)
from .normalize import Case, Document, parse_case
from .schema import DocumentFacts

log = logging.getLogger(__name__)

# An extractor reads one document (plus case context) and returns its facts.
Extractor = Callable[[Document, dict[str, Any]], DocumentFacts]

READY_EXPLANATION = (
    "READY: H&P current, signed surgical consent on file, required testing within window, "
    "anticoagulation management documented or not applicable, no acute safety exclusions"
)


def case_context(case: Case) -> dict[str, Any]:
    return {
        "procedure_type": case.procedure.type,
        "procedure_date": case.procedure.date.isoformat() if case.procedure.date else None,
        "medications": [m.name for m in case.medications],
    }


def _extract_one(
    doc: Document, context: dict[str, Any], extractors: Sequence[tuple[str, Extractor]]
) -> tuple[ExtractedDocument, Finding | None]:
    """Run extractors in order; fall through to the next on failure."""

    for name, extractor in extractors:
        try:
            facts, note = heuristic.reconcile_identity(doc, extractor(doc, context))
            if note:
                log.info("documents[%d] (%s): %s", doc.index, name, note)
            return ExtractedDocument(doc, facts, name), None
        except Exception as exc:  # noqa: BLE001 - any extractor failure falls through
            log.warning("extractor %s failed on documents[%d]: %s", name, doc.index, exc)
    failure = Finding(
        rule="extraction",
        status=Status.UNKNOWN,
        category=rules.MISSING,
        description=f"Could not read documents[{doc.index}]",
        source=f"documents[{doc.index}]",
        details=f"All extractors failed on documents[{doc.index}] ({doc.type})",
    )
    return ExtractedDocument(doc, DocumentFacts.empty(), "none"), failure


def extract_documents(
    case: Case, extractors: Sequence[tuple[str, Extractor]], max_workers: int | None = None
) -> tuple[list[ExtractedDocument], list[Finding]]:
    """Extract every document concurrently; results keep document order."""

    context = case_context(case)
    workers = max_workers or int(os.environ.get("TRIAGE_WORKERS", "4"))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(lambda doc: _extract_one(doc, context, extractors), case.documents))
    return [doc for doc, _ in results], [failure for _, failure in results if failure]


def evaluate(case: Case, docs: list[ExtractedDocument]) -> list[Finding]:
    return [
        *rules.check_required_fields(case),
        *rules.check_history_and_physical(case, docs),
        *rules.check_consent(case, docs),
        *rules.check_testing(case),
        *rules.check_anticoagulation(case, docs),
        *rules.check_acute_safety(case, docs),
        *tripwires.check(docs),
    ]


def decide(findings: list[Finding]) -> TriageOutput:
    blocked: dict[str, list[str]] = {}
    for f in findings:
        if f.status is Status.UNKNOWN and f.blocked_by:
            blocked.setdefault(f.blocked_by, []).append(f.description)

    issues = []
    for f in findings:
        if f.status is Status.PASS or (f.status is Status.UNKNOWN and f.blocked_by):
            continue
        category = f.category if f.status is Status.FAIL else rules.MISSING
        details = f.details
        if f.source in blocked:
            details += "; cannot evaluate: " + ", ".join(blocked[f.source])
        issues.append(TriageIssue(category=category, description=f.description,
                                  evidence=Evidence(source=f.source, details=details)))

    issues.sort(key=lambda i: (CATEGORY_ORDER.index(i.category), i.evidence.source, i.description))
    if any(i.category == rules.SAFETY for i in issues):
        decision = "NOT_CLEARED"
    elif issues:
        decision = "NEEDS_FOLLOW_UP"
    else:
        decision = "READY"
    explanation = " | ".join(f"{i.category}: {i.description}" for i in issues) or READY_EXPLANATION
    return TriageOutput(decision=decision, issues=issues, explanation=explanation)


def triage(
    submission: dict[str, Any],
    extractors: Sequence[tuple[str, Extractor]] = (("heuristic", heuristic.extract),),
) -> TriageOutput:
    case = parse_case(submission)
    docs, extraction_findings = extract_documents(case, extractors)
    return decide([*extraction_findings, *evaluate(case, docs)])
