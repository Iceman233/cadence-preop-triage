"""LLM document extractor (OpenAI Responses API, strict structured output).

The model only reads: it fills DocumentFacts for one document. It never sees the
policy and never makes a decision. Every claim is then verified in code against the
document text; a claim whose quote is not in the document is dropped or downgraded, so
an unsupported claim can never help a case reach READY.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

from .normalize import Document
from .schema import AnticoagPlan, DocumentFacts, HPFacts, PlanStep, VitalReading

log = logging.getLogger(__name__)

# Bump when the prompt or schema changes so cached extractions are not reused.
PROMPT_VERSION = "2026-10-02.4"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / ".cache" / "llm"

INSTRUCTIONS = """
You extract facts from ONE clinical document for a pre-operative scheduling review.
Report only what the document states. Do not judge whether the patient is ready for surgery.

Rules:
- Every quote must be ONE contiguous span copied verbatim from the document text (exact
  characters, no paraphrase). Never join fragments from different parts of a line or document.
- doc_kind: HISTORY_AND_PHYSICAL only for a document that is itself a history and physical;
  SURGICAL_CONSENT for a consent form or consent note; else OTHER. Pre-admission testing visits,
  nursing intakes, and anesthesia evaluations are OTHER even when they contain history or exam
  findings. A note that merely says "H&P reviewed" is OTHER.
- hp: fill only when doc_kind is HISTORY_AND_PHYSICAL.
- consent: fill only when doc_kind is SURGICAL_CONSENT. patient_signature describes the PATIENT's
  signature only. A physician attestation or physician e-signature is never the patient's.
  PENDING = patient has not signed yet / will return it; NOT_COMPLETED = e-signature request
  outstanding; ABSENT = blank signature line or form not signed; UNCLEAR = cannot tell.
- vitals: every blood pressure and temperature reading in the document, in document order, with
  time_of_day (HH:MM) when stated. One entry per measurement: a BLOOD_PRESSURE entry has null
  temperature fields and a TEMPERATURE entry has null systolic/diastolic, even when both appear on
  the same line; a line such as "BP 127/80, T 98.0 F" yields two entries. Abbreviated forms count
  ("T 98.9 F", "Temp: 37.1C", "98.6 F temporal"). quote = the shortest contiguous span containing
  the value (e.g. "T 98.0 F"); put the time only in time_of_day, never in the quote. Values exactly
  as written; do not convert units. "See flowsheet" or "deferred" is not a reading.
- medication_mentions: every medication the document says the patient is taking, has stopped, or
  may be taking (status UNCERTAIN). Do not include instructions ("hold X 2 days before surgery") as
  mentions. llm_drug_class: ANTICOAGULANT, ANTIPLATELET, OTHER, or UNKNOWN.
- anticoag_plans: one entry per anticoagulant whose perioperative management the document discusses
  by name or as "anticoagulation". Generic statements about home medications ("continue home
  medications except as directed") are not an anticoagulation plan.
  pre_op / post_op SPECIFIC = a concrete action with timing ("stop 5 days before", "last dose 03/15",
  "resume 24-48 hours after surgery", "restart on POD 1"). VAGUE = mentioned without a concrete action
  ("will be addressed after surgery"). ABSENT = not addressed. bridging_agents = drugs used to bridge
  (e.g. enoxaparin). defers_decision = management handed off ("to be determined by the surgical team",
  "recommendations to follow", "follow up with cardiology").
- Use empty lists and nulls when something is not present.
""".strip()


# Structured output mangles some non-ASCII symbols when the model copies them ("°C" has come
# back as "\x00b0C" and as "\x176"). The model therefore sees ASCII text, and quotes are
# matched back to the original document for verification and evidence.
_ASCII_FOR_MODEL = str.maketrans({"°": " ", "–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "\u00a0": " "})
_KEPT = re.compile(r"[a-z0-9./:]")


def for_model(text: str) -> str:
    return text.translate(_ASCII_FOR_MODEL)


def repair(text: str) -> str:
    """Drop control characters a model may emit inside a string."""

    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")


def _normalize(text: str) -> str:
    return "".join(_KEPT.findall(unicodedata.normalize("NFKC", repair(text or "")).lower()))


def locate(quote: str | None, text: str) -> str | None:
    """Return the exact original span of `quote` in `text`, or None.

    Matching uses letters, digits and / . : only, so every word and number must appear
    in order, while whitespace and symbols (degree signs, quotes) may differ. The span
    returned is the document's own wording, never the model's copy.
    """

    needle = _normalize(quote or "")
    if not needle:
        return None
    chars, index = [], []
    for i, ch in enumerate(text):
        for kept in _normalize(ch):
            chars.append(kept)
            index.append(i)
    pos = "".join(chars).find(needle)
    if pos < 0:
        return None
    start, end = index[pos], index[pos + len(needle) - 1] + 1
    # Keep closing punctuation the quote itself ends with: "(oral)" rather than "(oral".
    tail = re.search(r"[^\w\s]+$", (quote or "").strip())
    if tail and text.startswith(tail.group(0), end):
        end += len(tail.group(0))
    return text[start:end]


def _number_in(value: float | None, quote: str) -> bool:
    if value is None:
        return False
    rendered = f"{value:g}"
    return re.search(rf"(?<![\d.]){re.escape(rendered)}(?![\d])", quote) is not None


def _split_combined(readings: list[VitalReading]) -> list[VitalReading]:
    """A single entry carrying both a BP and a temperature becomes two readings."""

    out = []
    for r in readings:
        has_bp = r.systolic is not None and r.diastolic is not None
        has_temp = r.temp_value is not None
        if has_bp:
            out.append(r.model_copy(update={"kind": "BLOOD_PRESSURE", "temp_value": None, "temp_unit": None}))
        if has_temp:
            out.append(r.model_copy(update={"kind": "TEMPERATURE", "systolic": None, "diastolic": None}))
        if not has_bp and not has_temp:
            out.append(r)
    return out


def verify(facts: DocumentFacts, text: str) -> tuple[DocumentFacts, list[str]]:
    """Drop or downgrade claims whose quotes are not in the document; replace every kept
    quote with the document's exact wording. Returns the verified facts and notes."""

    notes: list[str] = []

    hp = facts.hp
    if hp:
        quote = locate(hp.purpose_quote, text)
        if hp.purpose != "UNCLEAR" and quote is None:
            notes.append("hp.purpose quote not found")
            hp = HPFacts(purpose="UNCLEAR", purpose_quote=None)
        else:
            hp = hp.model_copy(update={"purpose_quote": quote})

    consent = facts.consent
    if consent:
        procedure_quote = locate(consent.procedure_quote, text)
        signature_quote = locate(consent.patient_signature_quote, text)
        updates: dict[str, Any] = {"procedure_quote": procedure_quote, "patient_signature_quote": signature_quote}
        if consent.procedure_named and procedure_quote is None:
            notes.append("consent.procedure quote not found")
            # Unverified procedure name: treat as unnamed rather than guessing.
            updates["procedure_named"] = None
        if consent.patient_signature == "SIGNED" and signature_quote is None:
            notes.append("consent.patient_signature quote not found")
            updates.update(patient_signature="UNCLEAR", patient_signed_date=None)
        consent = consent.model_copy(update=updates)

    vitals = []
    for reading in _split_combined(facts.vitals):
        values = (reading.systolic, reading.diastolic) if reading.kind == "BLOOD_PRESSURE" else (reading.temp_value,)
        quote = locate(reading.quote, text)
        if quote is not None and all(_number_in(v, quote) for v in values):
            vitals.append(reading.model_copy(update={"quote": quote}))
        else:
            notes.append(f"vital {values} not supported by its quote")

    mentions = []
    for mention in facts.medication_mentions:
        quote = locate(mention.quote, text)
        if quote is not None and _normalize(mention.name_as_written) in _normalize(quote):
            mentions.append(mention.model_copy(update={"quote": quote}))
        else:
            notes.append(f"medication mention {mention.name_as_written!r} not supported by its quote")

    plans = []
    for plan in facts.anticoag_plans:

        def checked(step: PlanStep, label: str) -> PlanStep:
            quote = locate(step.quote, text)
            if step.status == "SPECIFIC" and quote is None:
                notes.append(f"plan {label} quote not found")
                return PlanStep(status="ABSENT", quote=None)
            return PlanStep(status=step.status, quote=quote)

        plans.append(
            AnticoagPlan(
                drug_as_written=locate(plan.drug_as_written, text),
                pre_op=checked(plan.pre_op, "pre_op"),
                post_op=checked(plan.post_op, "post_op"),
                bridging_agents=[a for a in (locate(b, text) for b in plan.bridging_agents) if a],
                # A deferral only makes the plan weaker, so it is kept even if unquoted.
                defers_decision=plan.defers_decision,
                defers_quote=locate(plan.defers_quote, text),
            )
        )

    verified = facts.model_copy(
        update={"hp": hp, "consent": consent, "vitals": vitals, "medication_mentions": mentions, "anticoag_plans": plans}
    )
    return verified, notes


class LLMExtractor:
    """Callable extractor: (Document, context) -> DocumentFacts."""

    def __init__(
        self,
        *,
        model: str,
        client: Any | None = None,
        cache_dir: Path | None = DEFAULT_CACHE_DIR,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=timeout, max_retries=max_retries)
        self.client = client
        self.model = model
        if os.environ.get("TRIAGE_LLM_CACHE", "1") == "0":
            cache_dir = None
        self.cache_dir = cache_dir

    def _cache_path(self, payload: str) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(f"{self.model}\n{PROMPT_VERSION}\n{payload}".encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _call(self, payload: str) -> DocumentFacts:
        response = self.client.responses.parse(
            model=self.model,
            instructions=INSTRUCTIONS,
            input=[{"role": "user", "content": payload}],
            text_format=DocumentFacts,
            temperature=0,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("model returned no parsed output (refusal or incomplete response)")
        return parsed

    def __call__(self, doc: Document, context: dict[str, Any]) -> DocumentFacts:
        payload = json.dumps(
            {
                "case_context": context,
                "document": {"type": doc.type, "date": doc.date_raw, "text": for_model(doc.text)},
            },
            sort_keys=True,
        )
        path = self._cache_path(payload)
        if path is not None and path.exists():
            raw = DocumentFacts.model_validate_json(path.read_text())
        else:
            raw = self._call(payload)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(raw.model_dump_json())
        facts, notes = verify(raw, doc.text)
        for note in notes:
            log.info("documents[%d] verification: %s", doc.index, note)
        return facts
