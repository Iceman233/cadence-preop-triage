"""Policy rules. Each check returns Findings with PASS / FAIL / UNKNOWN and exact evidence.

FAIL means the input is present but does not satisfy the policy (including documents
that are present but incomplete or ambiguous). UNKNOWN means an input the rule needs is
absent or could not be verified. The engine maps FAIL to the rule's category and
UNKNOWN to MISSING_REQUIRED_DATA.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date

from . import policy
from .models import ExtractedDocument, Finding, Status
from .normalize import BLOOD_PRESSURE, TEMPERATURE, Case, Timestamp, fahrenheit, parse_date

DOCS = "REQUIRED_DOCUMENTATION"
TESTING = "REQUIRED_TESTING"
ANTICOAG = "ANTICOAGULATION_MANAGEMENT"
SAFETY = "ACUTE_SAFETY_EXCLUSION"
MISSING = "MISSING_REQUIRED_DATA"

PROCEDURE_DATE = "procedure.procedure_date"
PROCEDURE_RISK = "procedure.procedure_risk"


def _json(value: object) -> str:
    return json.dumps(value, default=str)


def _quote(text: str | None, limit: int = 220) -> str:
    text = " ".join((text or "").split())
    return f'"{text[:limit]}{"..." if len(text) > limit else ""}"'


# -------------------------
# Required fields
# -------------------------


def check_required_fields(case: Case) -> list[Finding]:
    findings = []
    proc = case.procedure
    if proc.date is None:
        findings.append(
            Finding(
                rule="required_fields.procedure_date",
                status=Status.UNKNOWN,
                category=MISSING,
                description="Missing procedure date",
                source=PROCEDURE_DATE,
                details=f"{PROCEDURE_DATE} is {_json(proc.date_raw)}"
                + ("" if proc.date_raw is None else " (not a recognizable date)"),
            )
        )
    if proc.risk is None:
        valid = "/".join(policy.REQUIRED_TESTS)
        findings.append(
            Finding(
                rule="required_fields.procedure_risk",
                status=Status.UNKNOWN,
                category=MISSING,
                description="Missing procedure risk level",
                source=PROCEDURE_RISK,
                details=f"{PROCEDURE_RISK} is {_json(proc.risk_raw)}"
                + ("" if proc.risk_raw is None else f" (expected one of {valid})"),
            )
        )
    return findings


# -------------------------
# Rule 1: H&P and consent
# -------------------------

_DATE_OF_SERVICE_RE = re.compile(r"date of service:\s*(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})", re.I)


def _hp_date(item: ExtractedDocument) -> date | None:
    """The document date, or the in-text date of service if that is older (stricter)."""

    dates = [item.doc.date]
    stated = _DATE_OF_SERVICE_RE.search(item.doc.text)
    if stated:
        dates.append(parse_date(stated.group(1)))
    dates = [d for d in dates if d is not None]
    return min(dates) if dates else None


def check_history_and_physical(case: Case, docs: list[ExtractedDocument]) -> list[Finding]:
    rule = "rule1.history_and_physical"
    hps = [d for d in docs if d.facts.doc_kind == "HISTORY_AND_PHYSICAL"]
    if not hps:
        types = ", ".join(sorted({d.doc.type for d in docs if d.doc.type})) or "none"
        return [
            Finding(rule, Status.FAIL, DOCS, "Missing History and Physical (H&P)", "documents",
                    f"No H&P among {len(docs)} documents (document types: {types})")
        ]
    proc_date = case.procedure.date
    if proc_date is None:
        return [
            Finding(rule, Status.UNKNOWN, MISSING, "H&P recency", PROCEDURE_DATE,
                    "procedure date missing", blocked_by=PROCEDURE_DATE)
        ]

    dated = [(_hp_date(d), d) for d in hps]
    eligible = [(when, d) for when, d in dated if when is not None and when <= proc_date]
    if not eligible:
        undated = [d for when, d in dated if when is None]
        if undated:
            return [
                Finding(rule, Status.UNKNOWN, MISSING, "H&P date missing", f"{undated[0].source}.date",
                        f"{undated[0].label()} has no usable date")
            ]
        return [
            Finding(rule, Status.FAIL, DOCS, "No H&P completed before the procedure date", "documents",
                    "; ".join(f"{d.label()} is dated after procedure date {proc_date}" for _, d in dated))
        ]

    when, latest = max(eligible, key=lambda pair: (pair[0], pair[1].doc.index))
    days = (proc_date - when).days
    details = (
        f"Most recent H&P {latest.label()} dated {when} is {days} days before procedure date "
        f"{proc_date} (limit {policy.H_AND_P_WINDOW_DAYS})"
    )
    if days <= policy.H_AND_P_WINDOW_DAYS:
        return [Finding(rule, Status.PASS, DOCS, "H&P current", latest.source, details)]
    return [Finding(rule, Status.FAIL, DOCS, "H&P outdated (older than 30 days)", latest.source, details)]


_PROCEDURE_STOPWORDS = {"elective", "procedure", "surgery", "surgical", "the", "of", "for", "with", "and", "a"}


def _procedure_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _PROCEDURE_STOPWORDS}


def procedure_matches(named: str | None, scheduled: str | None) -> bool | None:
    """True/False when both are known; None when the consent names no procedure."""

    if not named or not scheduled:
        return None
    a, b = _procedure_tokens(named), _procedure_tokens(scheduled)
    if not a or not b:
        return None
    if a <= b or b <= a:
        return True
    return len(a & b) / len(a | b) >= 0.5


def check_consent(case: Case, docs: list[ExtractedDocument]) -> list[Finding]:
    rule = "rule1.surgical_consent"
    consents = [d for d in docs if d.facts.doc_kind == "SURGICAL_CONSENT" and d.facts.consent]
    if not consents:
        return [
            Finding(rule, Status.FAIL, DOCS, "Missing signed surgical consent", "documents",
                    f"No surgical consent among {len(docs)} documents")
        ]

    scheduled = case.procedure.type
    candidates = [d for d in consents if procedure_matches(d.facts.consent.procedure_named, scheduled) is not False]
    if not candidates:
        wrong = consents[-1]
        return [
            Finding(rule, Status.FAIL, DOCS, "Surgical consent is for a different procedure", wrong.source,
                    f"{wrong.label()} authorizes {_quote(wrong.facts.consent.procedure_named)} but the "
                    f"scheduled procedure is {_quote(scheduled)}")
        ]

    def recency(d: ExtractedDocument) -> tuple[date, int]:
        signed = parse_date(d.facts.consent.patient_signed_date)
        return (signed or d.doc.date or date.min, d.doc.index)

    latest = max(candidates, key=recency)
    consent = latest.facts.consent
    details = f"{latest.label()}: patient signature {consent.patient_signature}, {_quote(consent.patient_signature_quote)}"
    if consent.patient_signature == "SIGNED":
        return [Finding(rule, Status.PASS, DOCS, "Signed surgical consent on file", latest.source, details)]
    return [Finding(rule, Status.FAIL, DOCS, "Surgical consent not signed by patient", latest.source, details)]


# -------------------------
# Rule 2: testing
# -------------------------


def check_testing(case: Case) -> list[Finding]:
    rule = "rule2.required_testing"
    risk, proc_date = case.procedure.risk, case.procedure.date
    if risk is None:
        return [Finding(rule, Status.UNKNOWN, MISSING, "required test selection", PROCEDURE_RISK,
                        "procedure risk missing", blocked_by=PROCEDURE_RISK)]
    if proc_date is None:
        return [Finding(rule, Status.UNKNOWN, MISSING, "required test recency", PROCEDURE_DATE,
                        "procedure date missing", blocked_by=PROCEDURE_DATE)]

    findings = []
    for test, window in policy.REQUIRED_TESTS[risk].items():
        results = [lab for lab in case.labs if lab.test == test]
        excluded = [
            f"labs[{lab.index}] status {lab.status!r}"
            for lab in results
            if lab.status and lab.status not in policy.USABLE_LAB_STATUSES
        ]
        usable = [lab for lab in results if not lab.status or lab.status in policy.USABLE_LAB_STATUSES]
        dated = [lab for lab in usable if lab.effective is not None and lab.effective <= proc_date]
        future = [f"labs[{lab.index}] dated {lab.effective} is after the procedure" for lab in usable
                  if lab.effective is not None and lab.effective > proc_date]

        if not dated:
            undated = [lab for lab in usable if lab.effective is None]
            if undated:
                findings.append(Finding(rule, Status.UNKNOWN, MISSING, f"{test} result date missing",
                                        f"labs[{undated[0].index}].effective_at",
                                        f"labs[{undated[0].index}].effective_at is {undated[0].effective_raw!r}"))
                continue
            notes = "; ".join(excluded + future)
            present = ", ".join(sorted({lab.code or lab.display or "?" for lab in case.labs})) or "none"
            findings.append(Finding(rule, Status.FAIL, TESTING, f"Missing required {test} ({risk} risk)", "labs",
                                    f"No usable {test} result. Labs present: {present}" + (f". Excluded: {notes}" if notes else "")))
            continue

        latest = max(dated, key=lambda lab: (lab.effective, lab.index))
        days = (proc_date - latest.effective).days
        details = (
            f"Most recent usable {test} labs[{latest.index}] ({latest.display or latest.code}) effective "
            f"{latest.effective_raw} is {days} days before procedure date {proc_date} (limit {window}, {risk} risk)"
        )
        if excluded:
            details += f". Excluded: {'; '.join(excluded)}"
        if days <= window:
            findings.append(Finding(rule, Status.PASS, TESTING, f"{test} current", f"labs[{latest.index}]", details))
        else:
            findings.append(Finding(rule, Status.FAIL, TESTING, f"{test} outside {window}-day window",
                                    f"labs[{latest.index}]", details))
    return findings


# -------------------------
# Rule 3: anticoagulation
# -------------------------


@dataclass(frozen=True)
class _StatusEvidence:
    status: str  # TAKING / STOPPED / UNCERTAIN
    order: tuple
    source: str
    details: str


def _mention_generic(name: str, llm_class: str) -> str | None:
    generic, drug_class = policy.classify_drug(name)
    if generic:
        return generic if drug_class == policy.ANTICOAGULANT else None
    # Unlisted drug: fall back to the extractor's classification, keyed by the name.
    return name.strip().lower() if llm_class == "ANTICOAGULANT" else None


def _plan_generic(name: str | None) -> str | None:
    if not name:
        return None
    generic, _ = policy.classify_drug(name)
    return generic or name.strip().lower()


def check_anticoagulation(case: Case, docs: list[ExtractedDocument]) -> list[Finding]:
    rule = "rule3.anticoagulation"
    structured: dict[str, list] = {}
    for med in case.medications:
        if med.drug_class == policy.ANTICOAGULANT:
            structured.setdefault(med.generic, []).append(med)

    notes: dict[str, list[_StatusEvidence]] = {}
    for item in docs:
        for mention in item.facts.medication_mentions:
            generic = _mention_generic(mention.name_as_written, mention.llm_drug_class)
            if not generic:
                continue
            position = item.doc.text.find(mention.quote)
            notes.setdefault(generic, []).append(
                _StatusEvidence(
                    status=mention.status,
                    order=(item.doc.date or date.min, item.doc.index, position),
                    source=item.source,
                    details=f"{item.label()}: {_quote(mention.quote)}",
                )
            )

    findings = []
    for generic in sorted(set(structured) | set(notes)):
        meds = structured.get(generic, [])
        latest_note = max(notes.get(generic, []), key=lambda e: e.order, default=None)
        note_status = latest_note.status if latest_note else None
        flags = {m.active for m in meds}
        listed = True if True in flags else (None if None in flags and meds else (False if meds else "absent"))

        evidence = [f"medications[{m.index}] {m.name!r} active={m.active_raw!r}" for m in meds]
        if latest_note:
            evidence.append(f"latest note {latest_note.details}")
        source = latest_note.source if latest_note and not meds else (f"medications[{meds[0].index}]" if meds else "documents")

        if note_status == "UNCERTAIN" or (listed is None and note_status is None) or (listed is True and note_status == "STOPPED"):
            findings.append(Finding(rule, Status.UNKNOWN, MISSING, f"Unconfirmed whether patient is taking {generic}",
                                    source, "; ".join(evidence)))
            continue
        active = (listed is True and note_status not in {"STOPPED", "UNCERTAIN"}) or note_status == "TAKING"
        if not active:
            findings.append(Finding(rule, Status.PASS, ANTICOAG, f"{generic} not active", source, "; ".join(evidence)))
            continue
        findings.append(_check_plan(rule, generic, source, evidence, docs))
    return findings


def _check_plan(rule: str, generic: str, source: str, evidence: list[str], docs: list[ExtractedDocument]) -> Finding:
    active_desc = f"Active anticoagulant {generic} ({'; '.join(evidence)})"
    plans = [
        (item, plan)
        for item in docs
        for plan in item.facts.anticoag_plans
        if _plan_generic(plan.drug_as_written) == generic
    ]
    if not plans:
        others = [
            f"{item.label()} addresses {plan.drug_as_written or 'an unnamed drug'}"
            for item in docs
            for plan in item.facts.anticoag_plans
        ]
        details = f"{active_desc} but no perioperative plan for {generic} found"
        if others:
            details += f"; other plans: {'; '.join(others)}"
        return Finding(rule, Status.FAIL, ANTICOAG, f"Missing perioperative anticoagulation plan for {generic}",
                       source, details)

    # The most recent plan governs (a newer vague note reopens the question). Same-day
    # plans tie-break on completeness, not on document order.
    def recency(pair) -> tuple:
        item, plan = pair
        completeness = sum(step.status == "SPECIFIC" for step in (plan.pre_op, plan.post_op))
        return (item.doc.date or date.min, completeness, item.doc.index)

    item, plan = max(plans, key=recency)
    parts = [
        f"pre-op {plan.pre_op.status}" + (f" {_quote(plan.pre_op.quote)}" if plan.pre_op.quote else ""),
        f"post-op {plan.post_op.status}" + (f" {_quote(plan.post_op.quote)}" if plan.post_op.quote else ""),
    ]
    if plan.defers_decision:
        parts.append(f"defers decision {_quote(plan.defers_quote)}")
    details = f"{item.label()} plan for {generic}: " + "; ".join(parts)
    if plan.pre_op.status == "SPECIFIC" and plan.post_op.status == "SPECIFIC":
        return Finding(rule, Status.PASS, ANTICOAG, f"Perioperative plan for {generic} documented", item.source, details)
    missing = [side for side, step in (("pre-operative", plan.pre_op), ("post-operative", plan.post_op))
               if step.status != "SPECIFIC"]
    return Finding(rule, Status.FAIL, ANTICOAG,
                   f"Perioperative plan for {generic} is incomplete ({' and '.join(missing)} management not specified)",
                   item.source, details)


# -------------------------
# Rule 4: acute safety
# -------------------------


@dataclass(frozen=True)
class _Reading:
    kind: str
    order: tuple
    systolic: float | None
    diastolic: float | None
    temp_f: float | None
    source: str
    details: str

    def severity(self) -> float:
        if self.kind == BLOOD_PRESSURE:
            return max(self.systolic / policy.SYSTOLIC_EXCLUSION_MMHG, self.diastolic / policy.DIASTOLIC_EXCLUSION_MMHG)
        return self.temp_f


def _minutes(time_of_day: str | None) -> int | None:
    match = re.match(r"^\s*(\d{1,2}):(\d{2})", time_of_day or "")
    return int(match.group(1)) * 60 + int(match.group(2)) if match else None


_END_OF_DAY = 24 * 60


def collect_readings(case: Case, docs: list[ExtractedDocument]) -> list[_Reading]:
    """Structured and note vitals on one timeline.

    Ordered by date, then time of day; a note reading without a time sorts after
    same-day timed readings (end of day). Readings after the review time are ignored.
    """

    review_day = case.review_at.day if case.review_at else None
    readings = []
    for vital in case.vitals:
        if vital.at is None or (review_day and vital.at.day > review_day):
            continue
        minutes = vital.at.minutes if vital.at.minutes is not None else _END_OF_DAY
        order = (vital.at.day, minutes, 0, vital.index)
        if vital.kind == BLOOD_PRESSURE and vital.systolic is not None and vital.diastolic is not None:
            readings.append(_Reading(BLOOD_PRESSURE, order, vital.systolic, vital.diastolic, None, f"vitals[{vital.index}]",
                                     f"vitals[{vital.index}] {vital.systolic:g}/{vital.diastolic:g} mmHg at {vital.at}"))
        elif vital.kind == TEMPERATURE and vital.temp_f is not None:
            readings.append(_Reading(TEMPERATURE, order, None, None, vital.temp_f, f"vitals[{vital.index}]",
                                     f"vitals[{vital.index}] {vital.temp_raw} ({vital.temp_f:g} F) at {vital.at}"))

    for item in docs:
        day = item.doc.date
        if day is None or (review_day and day > review_day):
            continue
        for seq, reading in enumerate(item.facts.vitals):
            minutes = _minutes(reading.time_of_day)
            order = (day, minutes if minutes is not None else _END_OF_DAY, 1, item.doc.index, seq)
            when = Timestamp(day, minutes)
            if reading.kind == "BLOOD_PRESSURE" and reading.systolic is not None and reading.diastolic is not None:
                readings.append(_Reading(BLOOD_PRESSURE, order, reading.systolic, reading.diastolic, None, item.source,
                                         f"{item.label()} {reading.systolic:g}/{reading.diastolic:g} mmHg at {when}: {_quote(reading.quote)}"))
            elif reading.kind == "TEMPERATURE" and reading.temp_value is not None:
                temp_f = fahrenheit(reading.temp_value, reading.temp_unit)
                written = f"{reading.temp_value:g} {reading.temp_unit or '(no unit)'}"
                readings.append(_Reading(TEMPERATURE, order, None, None, temp_f, item.source,
                                         f"{item.label()} {written} ({temp_f:g} F) at {when}: {_quote(reading.quote)}"))
    return readings


def _latest(readings: list[_Reading]) -> _Reading:
    """Most recent reading; on an exact tie the more severe one wins."""

    top = max(r.order[:2] for r in readings)
    return max((r for r in readings if r.order[:2] == top), key=lambda r: (r.severity(), r.order))


def check_acute_safety(case: Case, docs: list[ExtractedDocument]) -> list[Finding]:
    rule = "rule4.acute_safety"
    readings = collect_readings(case, docs)
    findings = []

    bps = [r for r in readings if r.kind == BLOOD_PRESSURE]
    if not bps:
        findings.append(Finding(rule, Status.UNKNOWN, MISSING, "Missing blood pressure", "vitals",
                                "No blood pressure reading in vitals or documents"))
    else:
        bp = _latest(bps)
        if bp.systolic >= policy.SYSTOLIC_EXCLUSION_MMHG:
            findings.append(Finding(rule, Status.FAIL, SAFETY,
                                    f"Systolic blood pressure >= {policy.SYSTOLIC_EXCLUSION_MMHG} mmHg", bp.source,
                                    f"Most recent BP: {bp.details}"))
        if bp.diastolic >= policy.DIASTOLIC_EXCLUSION_MMHG:
            findings.append(Finding(rule, Status.FAIL, SAFETY,
                                    f"Diastolic blood pressure >= {policy.DIASTOLIC_EXCLUSION_MMHG} mmHg", bp.source,
                                    f"Most recent BP: {bp.details}"))
        if bp.systolic < policy.SYSTOLIC_EXCLUSION_MMHG and bp.diastolic < policy.DIASTOLIC_EXCLUSION_MMHG:
            findings.append(Finding(rule, Status.PASS, SAFETY, "Blood pressure below exclusion thresholds", bp.source,
                                    f"Most recent BP: {bp.details}"))

    temps = [r for r in readings if r.kind == TEMPERATURE]
    if not temps:
        findings.append(Finding(rule, Status.UNKNOWN, MISSING, "Missing temperature", "vitals",
                                "No temperature reading in vitals or documents"))
    else:
        temp = _latest(temps)
        if temp.temp_f > policy.TEMPERATURE_EXCLUSION_F:
            findings.append(Finding(rule, Status.FAIL, SAFETY, f"Temperature > {policy.TEMPERATURE_EXCLUSION_F} F",
                                    temp.source, f"Most recent temperature: {temp.details}"))
        else:
            findings.append(Finding(rule, Status.PASS, SAFETY, "Temperature below exclusion threshold", temp.source,
                                    f"Most recent temperature: {temp.details}"))
    return findings
