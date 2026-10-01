"""Safety net against silent extraction misses.

Cheap, deterministic scans of each document's raw text for signals that would change
the decision (a blood pressure, a temperature, an anticoagulant name, an unsigned
consent). Any signal no extracted fact accounts for becomes an UNKNOWN finding, so a
missed fact can push a case to NEEDS_FOLLOW_UP but never silently to READY.
"""

from __future__ import annotations

import re

from . import heuristic, policy
from .models import ExtractedDocument, Finding, Status

MISSING = "MISSING_REQUIRED_DATA"

_BP_WORDS = re.compile(r"\bN?IBP\b|\bBP\b|blood pressure|mmHg|systolic|diastolic", re.I)
_BP_PAIR = re.compile(r"(?<![\d/.])(\d{2,3})\s*(?:/|over)\s*(\d{2,3})(?![\d/])", re.I)
_TEMP_WORDS = re.compile(r"temp|febrile|fever|°|degrees|\bT\s*\d", re.I)
# A decimal value, or an integer followed by a unit: "38.6", "101 F", "99 degrees".
_TEMP_VALUE = re.compile(
    r"(?<![\d./])(\d{2,3}\.\d+)(?![\d/%])|(?<![\d./])(\d{2,3})\s*(?:°|degrees|[CF]\b)", re.I
)
_UNSIGNED_RE = re.compile(
    r"signature:\s*(?:pending|_{3,})|not completed|not (?:yet )?signed|unsigned|no signature|pending signature", re.I
)


def _excerpt(text: str, start: int, end: int) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line = text[line_start : line_end if line_end != -1 else len(text)]
    return '"' + " ".join(line.split())[:200] + '"'


def check_document(item: ExtractedDocument) -> list[Finding]:
    text, facts = item.doc.text, item.facts
    findings = []

    def unverified(what: str, rule: str, details: str) -> None:
        findings.append(
            Finding(
                rule=f"tripwire.{rule}",
                status=Status.UNKNOWN,
                category=MISSING,
                description=f"Unverified {what} in {item.source}",
                source=item.source,
                details=f"{item.label()} mentions {details} that no extracted fact accounts for",
            )
        )

    # Deliberately broader than any extractor: an extractor's blind spots must not be
    # the tripwire's blind spots. Any BP-like pair or temperature-like value on a line
    # that talks about vitals has to be accounted for by an extracted reading.
    extracted_bp = {(v.systolic, v.diastolic) for v in facts.vitals if v.kind == "BLOOD_PRESSURE"}
    extracted_temp = {round(v.temp_value, 1) for v in facts.vitals if v.kind == "TEMPERATURE" and v.temp_value is not None}
    seen: set = set()
    for line_match in re.finditer(r"[^\n]+", text):
        line = line_match.group(0)
        where = _excerpt(text, line_match.start(), line_match.end())
        if _BP_WORDS.search(line):
            for m in _BP_PAIR.finditer(line):
                value = (float(m.group(1)), float(m.group(2)))
                if heuristic.plausible_bp(*value) and value not in extracted_bp and value not in seen:
                    seen.add(value)
                    unverified("blood pressure", "blood_pressure", f"{m.group(0)!r} in {where}")
        if _TEMP_WORDS.search(line):
            for m in _TEMP_VALUE.finditer(line):
                value = round(float(m.group(1) or m.group(2)), 1)
                plausible = 30 <= value <= 45 or 85 <= value <= 115
                if plausible and value not in extracted_temp and ("temp", value) not in seen:
                    seen.add(("temp", value))
                    unverified("temperature", "temperature", f"{m.group(0).strip()!r} in {where}")

    accounted = {policy.classify_drug(m.name_as_written)[0] for m in facts.medication_mentions}
    for plan in facts.anticoag_plans:
        accounted.add(policy.classify_drug(plan.drug_as_written or "")[0])
        accounted.update(policy.classify_drug(agent)[0] for agent in plan.bridging_agents)
    for generic, match in policy.find_drugs(text, drug_class=policy.ANTICOAGULANT):
        if generic not in accounted:
            accounted.add(generic)
            unverified("anticoagulant mention", "anticoagulant", f"{match.group(0)} in {_excerpt(text, match.start(), match.end())}")

    if facts.consent and facts.consent.patient_signature == "SIGNED":
        hit = _UNSIGNED_RE.search(text)
        if hit:
            unverified("consent signature", "consent_signature", f"{_excerpt(text, hit.start(), hit.end())} although the signature was read as SIGNED")

    return findings


def check(docs: list[ExtractedDocument]) -> list[Finding]:
    return [finding for item in docs for finding in check_document(item)]
