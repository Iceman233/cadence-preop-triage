"""Rule-based document extractor.

Fills the same DocumentFacts contract as the LLM extractor, so the two are
interchangeable. It is the offline fallback and the rules-only baseline for the
ablation. Every quote it emits is a verbatim line or sentence of the document.
"""

from __future__ import annotations

import re

from . import policy
from .normalize import Document
from .schema import (
    AnticoagPlan,
    ConsentFacts,
    DocumentFacts,
    HPFacts,
    MedicationMention,
    PlanStep,
    VitalReading,
)

# -------------------------
# Shared patterns
# -------------------------

HP_TYPE_RE = re.compile(
    r"\bh\s*&\s*p\b|\bh and p\b|history\s*(?:and|&)\s*(?:physical|pyhsical)", re.I
)
HP_HEADING_RE = re.compile(r"history\s*(?:and|&)\s*physical", re.I)
CONSENT_RE = re.compile(r"consent", re.I)

# "08:30", including "at 08:30: BP ..." (trailing colon), but not "08:30:15".
TIME_RE = re.compile(r"(?<![\d/:])([01]?\d|2[0-3]):([0-5]\d)(?!\d|:\d)")
BP_RE = re.compile(r"(?<![\d/.])(\d{2,3})\s*/\s*(\d{2,3})(?![\d/])")
BP_CONTEXT_RE = re.compile(r"\bN?IBP\b|\bBP\b|blood pressure|mmHg|vitals", re.I)
TEMP_RE = re.compile(r"(?<![\d.])(\d{2,3}(?:\.\d+)?)\s*°?\s*([CF])\b")

US_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")


def sentences(text: str) -> list[str]:
    """Split into lines, then sentences/clauses; each piece is a verbatim substring."""

    pieces = []
    for line in text.split("\n"):
        for piece in re.split(r"(?<=[.;])\s+", line):
            piece = piece.strip()
            if piece:
                pieces.append(piece)
    return pieces


def plausible_bp(systolic: float, diastolic: float) -> bool:
    return 60 <= systolic <= 300 and 30 <= diastolic <= 200 and systolic > diastolic


def plausible_temp(value: float, unit: str) -> bool:
    return 30 <= value <= 45 if unit.upper() == "C" else 85 <= value <= 115


# -------------------------
# Classification
# -------------------------


def classify(doc: Document) -> str:
    head = doc.text[:120]
    if CONSENT_RE.search(doc.type):
        return "SURGICAL_CONSENT"
    if HP_TYPE_RE.search(doc.type):
        return "HISTORY_AND_PHYSICAL"
    if re.search(r"informed consent", head, re.I):
        return "SURGICAL_CONSENT"
    if HP_HEADING_RE.search(head):
        return "HISTORY_AND_PHYSICAL"
    return "OTHER"


_PREOP_RE = re.compile(r"pre-?operative evaluation|pre-?op evaluation|prior to (?:planned )?\w+", re.I)
_NON_PREOP_RE = re.compile(r"wellness|establish(?:ing)? (?:primary )?care|longitudinal|retained for", re.I)


def hp_facts(doc: Document) -> HPFacts:
    for purpose, pattern in (("PREOP_EVALUATION", _PREOP_RE), ("NON_PREOP", _NON_PREOP_RE)):
        for piece in sentences(doc.text):
            if pattern.search(piece):
                return HPFacts(purpose=purpose, purpose_quote=piece)
    return HPFacts(purpose="UNCLEAR", purpose_quote=None)


# -------------------------
# Consent
# -------------------------

# Checked in order; negatives first so an unsigned form can never read as signed.
_SIGNATURE_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("NOT_COMPLETED", re.compile(r"not completed", re.I)),
    ("PENDING", re.compile(r"signature:\s*pending|will return the signed|additional time to review", re.I)),
    ("ABSENT", re.compile(r"signature:\s*_{3,}|not yet signed|printed for (?:patient )?review|signature line (?:is )?blank", re.I)),
    (
        "SIGNED",
        re.compile(
            r"patient signature on file|signed in clinic|signed by (?:the )?patient|"
            r"patient signature:\s*/s/|(?:e-?|electronically )signed by (?:the )?patient",
            re.I,
        ),
    ),
)
_ATTESTATION_RE = re.compile(r"physician attestation|^electronically signed by (?!.*patient)", re.I)


def consent_facts(doc: Document) -> ConsentFacts:
    procedure = re.search(r"^\s*Procedure:\s*(.+?)\s*$", doc.text, re.M)
    lines = [ln.strip() for ln in doc.text.split("\n") if ln.strip()]
    lines = [ln for ln in lines if not _ATTESTATION_RE.search(ln)]
    signature, quote, signed_date = "UNCLEAR", None, None
    for status, pattern in _SIGNATURE_CUES:
        hit = next((ln for ln in lines if pattern.search(ln)), None)
        if hit:
            signature, quote = status, hit
            if status == "SIGNED":
                date_match = US_DATE_RE.search(hit)
                signed_date = date_match.group(0) if date_match else None
            break
    return ConsentFacts(
        procedure_named=procedure.group(1) if procedure else None,
        procedure_quote=procedure.group(0).strip() if procedure else None,
        patient_signature=signature,
        patient_signature_quote=quote,
        patient_signed_date=signed_date,
    )


# -------------------------
# Vitals
# -------------------------


def vital_readings(text: str) -> list[VitalReading]:
    readings: list[VitalReading] = []
    for line in text.split("\n"):
        times = [(m.start(), f"{int(m.group(1)):02d}:{m.group(2)}") for m in TIME_RE.finditer(line)]

        def time_before(pos: int) -> str | None:
            earlier = [t for start, t in times if start < pos]
            return earlier[-1] if earlier else None

        found: list[tuple[int, VitalReading]] = []
        if BP_CONTEXT_RE.search(line):
            for m in BP_RE.finditer(line):
                systolic, diastolic = float(m.group(1)), float(m.group(2))
                if plausible_bp(systolic, diastolic):
                    found.append(
                        (
                            m.start(),
                            VitalReading(
                                kind="BLOOD_PRESSURE",
                                systolic=systolic,
                                diastolic=diastolic,
                                temp_value=None,
                                temp_unit=None,
                                time_of_day=time_before(m.start()),
                                quote=line.strip(),
                            ),
                        )
                    )
        for m in TEMP_RE.finditer(line):
            value, unit = float(m.group(1)), m.group(2).upper()
            if plausible_temp(value, unit):
                found.append(
                    (
                        m.start(),
                        VitalReading(
                            kind="TEMPERATURE",
                            systolic=None,
                            diastolic=None,
                            temp_value=value,
                            temp_unit=unit,
                            time_of_day=time_before(m.start()),
                            quote=line.strip(),
                        ),
                    )
                )
        readings.extend(reading for _, reading in sorted(found, key=lambda item: item[0]))
    return readings


# -------------------------
# Medications and anticoagulation plans
# -------------------------

_UNCERTAIN_RE = re.compile(r"unable to confirm|not sure|unsure|uncertain|unclear whether", re.I)
_STOPPED_RE = re.compile(r"discontinu|stopped|no longer|completed \d+ months? of", re.I)
_TAKING_RE = re.compile(r"taking|takes|maintained on|^\s*-\s*", re.I)
_SWITCH_TO_RE = re.compile(r"(?:replaced with|switched to|changed to|transitioned to)\s+(\w+)", re.I)
# Instruction sentences belong to plans, not to medication status.
_INSTRUCTION_RE = re.compile(
    r"\b(?:hold|resume|restart|bridg\w*|interrupt\w*|periprocedural|last dose)\b|before surgery|post-?op", re.I
)


def medication_mentions(text: str) -> list[MedicationMention]:
    mentions = []
    for piece in sentences(text):
        drugs = policy.find_drugs(piece)
        if not drugs or _INSTRUCTION_RE.search(piece):
            continue
        switched_to = {
            policy.classify_drug(m.group(1))[0] for m in _SWITCH_TO_RE.finditer(piece)
        } - {None}
        for generic, match in drugs:
            if generic in switched_to:
                status = "TAKING"
            elif _UNCERTAIN_RE.search(piece):
                status = "UNCERTAIN"
            elif _STOPPED_RE.search(piece) or switched_to:
                status = "STOPPED"
            elif _TAKING_RE.search(piece):
                status = "TAKING"
            else:
                continue
            mentions.append(
                MedicationMention(
                    name_as_written=match.group(0),
                    status=status,
                    llm_drug_class=policy.DRUGS[generic][0],
                    quote=piece,
                )
            )
    return mentions


_PRE_VERB_RE = re.compile(r"\b(?:hold|stop|discontinue|interrupt\w*)\b|last dose", re.I)
_PRE_TIMING_RE = re.compile(r"before|prior|pre-?op|last dose \d", re.I)
_POST_VERB_RE = re.compile(r"\b(?:resume|restart|re-start|reinitiate)\w*", re.I)
_POST_TIMING_RE = re.compile(r"hours?|\bday\b|\bPOD\b|postoperative day|post-?op|after surgery|evening of surgery", re.I)
_POST_VAGUE_RE = re.compile(r"post-?operative anticoagulation.*(?:addressed|determined|decided)", re.I)
_DEFERS_RE = re.compile(
    r"to be determined by|up to the surgical team|deferred to|recommendations? (?:will )?follow|"
    r"follow up with \w+ for .*recommendations|discuss (?:holding|interruption)",
    re.I,
)
_RANK = {"ABSENT": 0, "VAGUE": 1, "SPECIFIC": 2}


def anticoag_plans(text: str) -> list[AnticoagPlan]:
    plans: dict[str, dict] = {}

    def upgrade(plan: dict, key: str, status: str, quote: str) -> None:
        if _RANK[status] > _RANK[plan[key].status]:
            plan[key] = PlanStep(status=status, quote=quote)

    for line in text.split("\n"):
        # A drugless sentence ("Post-operative anticoagulation will be addressed after
        # surgery.") refers back to the latest non-bridging drug named on the same line.
        # Back-references never cross lines, where an unlisted drug ("hold metformin")
        # could be mistaken for the last anticoagulant named.
        last_drug: tuple[str, str, str] | None = None  # (generic, class, as written)
        for piece in sentences(line):
            drugs = policy.find_drugs(piece)
            primary = [
                (g, m.group(0))
                for g, m in drugs
                if policy.DRUGS[g][0] == policy.ANTICOAGULANT and g not in policy.BRIDGING_AGENTS
            ]
            bridging = [m.group(0) for g, m in drugs if g in policy.BRIDGING_AGENTS and re.search(r"bridg", piece, re.I)]
            subjects = [(g, m.group(0)) for g, m in drugs if g not in policy.BRIDGING_AGENTS]
            if subjects:
                generic, written = primary[-1] if primary else subjects[-1]
                last_drug = (generic, policy.DRUGS[generic][0], written)

            pre = post = None
            if _PRE_VERB_RE.search(piece):
                pre = "SPECIFIC" if _PRE_TIMING_RE.search(piece) else "VAGUE"
            if _POST_VERB_RE.search(piece):
                post = "SPECIFIC" if _POST_TIMING_RE.search(piece) else "VAGUE"
            elif _POST_VAGUE_RE.search(piece):
                post = "VAGUE"
            defers = bool(_DEFERS_RE.search(piece))
            if not (pre or post or defers or bridging):
                continue

            targets = primary
            if not targets and last_drug and last_drug[1] == policy.ANTICOAGULANT:
                targets = [(last_drug[0], last_drug[2])]
            # A bridging-only sentence ("Bridge with enoxaparin ... last dose 24 hours
            # pre-op") describes the bridge, not the primary drug's hold/resume.
            steps_apply = bool(primary) or not bridging
            for generic, written in targets:
                plan = plans.setdefault(
                    generic,
                    {
                        "drug_as_written": written,
                        "pre_op": PlanStep(status="ABSENT", quote=None),
                        "post_op": PlanStep(status="ABSENT", quote=None),
                        "bridging_agents": [],
                        "defers_decision": False,
                        "defers_quote": None,
                    },
                )
                if pre and steps_apply:
                    upgrade(plan, "pre_op", pre, piece)
                if post and steps_apply:
                    upgrade(plan, "post_op", post, piece)
                if defers and not plan["defers_decision"]:
                    plan["defers_decision"], plan["defers_quote"] = True, piece
                for agent in bridging:
                    if agent not in plan["bridging_agents"]:
                        plan["bridging_agents"].append(agent)

    return [AnticoagPlan(**plan) for plan in plans.values()]


# -------------------------
# Entry point
# -------------------------


def extract(doc: Document, context: dict | None = None) -> DocumentFacts:
    kind = classify(doc)
    return DocumentFacts(
        doc_kind=kind,
        hp=hp_facts(doc) if kind == "HISTORY_AND_PHYSICAL" else None,
        consent=consent_facts(doc) if kind == "SURGICAL_CONSENT" else None,
        vitals=vital_readings(doc.text),
        medication_mentions=medication_mentions(doc.text),
        anticoag_plans=anticoag_plans(doc.text),
    )
