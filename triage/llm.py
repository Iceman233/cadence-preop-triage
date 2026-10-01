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
from pathlib import Path
from typing import Any

from .normalize import Document
from .schema import AnticoagPlan, DocumentFacts, HPFacts, PlanStep

log = logging.getLogger(__name__)

# Bump when the prompt or schema changes so cached extractions are not reused.
PROMPT_VERSION = "2026-10-01.1"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / ".cache" / "llm"

INSTRUCTIONS = """
You extract facts from ONE clinical document for a pre-operative scheduling review.
Report only what the document states. Do not judge whether the patient is ready for surgery.

Rules:
- Every quote must be copied verbatim from the document text (exact characters, no paraphrase).
- doc_kind: decide from the content, not only the title. HISTORY_AND_PHYSICAL for a history and
  physical exam note (any purpose); SURGICAL_CONSENT for a consent form or consent note; else OTHER.
  A note that merely says "H&P reviewed" is OTHER.
- hp: fill only when doc_kind is HISTORY_AND_PHYSICAL.
- consent: fill only when doc_kind is SURGICAL_CONSENT. patient_signature describes the PATIENT's
  signature only. A physician attestation or physician e-signature is never the patient's.
  PENDING = patient has not signed yet / will return it; NOT_COMPLETED = e-signature request
  outstanding; ABSENT = blank signature line or form not signed; UNCLEAR = cannot tell.
- vitals: every blood pressure and temperature reading in the document, in document order, with
  time_of_day (HH:MM) when stated. Values exactly as written; do not convert units.
  "See flowsheet" or "deferred" is not a reading.
- medication_mentions: every medication the document says the patient is taking, has stopped, or
  may be taking (status UNCERTAIN). Do not include instructions ("hold X 2 days before surgery") as
  mentions. llm_drug_class: ANTICOAGULANT, ANTIPLATELET, OTHER, or UNKNOWN.
- anticoag_plans: one entry per anticoagulant whose perioperative management the document discusses.
  pre_op / post_op SPECIFIC = a concrete action with timing ("stop 5 days before", "last dose 03/15",
  "resume 24-48 hours after surgery", "restart on POD 1"). VAGUE = mentioned without a concrete action
  ("will be addressed after surgery"). ABSENT = not addressed. bridging_agents = drugs used to bridge
  (e.g. enoxaparin). defers_decision = management handed off ("to be determined by the surgical team",
  "recommendations to follow", "follow up with cardiology").
- Use empty lists and nulls when something is not present.
""".strip()


def _normalize(text: str) -> str:
    return " ".join((text or "").split()).lower()


def _in_text(quote: str | None, text: str) -> bool:
    return bool(quote) and _normalize(quote) in _normalize(text)


def _number_in(value: float | None, quote: str) -> bool:
    if value is None:
        return False
    rendered = f"{value:g}"
    return re.search(rf"(?<![\d.]){re.escape(rendered)}(?![\d])", quote) is not None


def verify(facts: DocumentFacts, text: str) -> tuple[DocumentFacts, list[str]]:
    """Drop or downgrade claims whose quotes are not in the document. Returns notes."""

    notes: list[str] = []

    hp = facts.hp
    if hp and hp.purpose != "UNCLEAR" and not _in_text(hp.purpose_quote, text):
        notes.append("hp.purpose quote not found")
        hp = HPFacts(purpose="UNCLEAR", purpose_quote=None)

    consent = facts.consent
    if consent:
        updates: dict[str, Any] = {}
        if consent.procedure_named and not _in_text(consent.procedure_quote, text):
            notes.append("consent.procedure quote not found")
            # Unverified procedure name: treat as unnamed rather than guessing.
            updates.update(procedure_named=None, procedure_quote=None)
        if consent.patient_signature == "SIGNED" and not _in_text(consent.patient_signature_quote, text):
            notes.append("consent.patient_signature quote not found")
            updates.update(patient_signature="UNCLEAR", patient_signed_date=None)
        consent = consent.model_copy(update=updates) if updates else consent

    vitals = []
    for reading in facts.vitals:
        values = (reading.systolic, reading.diastolic) if reading.kind == "BLOOD_PRESSURE" else (reading.temp_value,)
        if _in_text(reading.quote, text) and all(_number_in(v, reading.quote) for v in values):
            vitals.append(reading)
        else:
            notes.append(f"vital {values} not supported by its quote")

    mentions = []
    for mention in facts.medication_mentions:
        if _in_text(mention.quote, text) and _normalize(mention.name_as_written) in _normalize(mention.quote):
            mentions.append(mention)
        else:
            notes.append(f"medication mention {mention.name_as_written!r} not supported by its quote")

    plans = []
    for plan in facts.anticoag_plans:
        drug = plan.drug_as_written if plan.drug_as_written and _in_text(plan.drug_as_written, text) else None

        def checked(step: PlanStep, label: str) -> PlanStep:
            if step.status == "SPECIFIC" and not _in_text(step.quote, text):
                notes.append(f"plan {label} quote not found")
                return PlanStep(status="ABSENT", quote=None)
            return step

        plans.append(
            AnticoagPlan(
                drug_as_written=drug,
                pre_op=checked(plan.pre_op, "pre_op"),
                post_op=checked(plan.post_op, "post_op"),
                bridging_agents=[a for a in plan.bridging_agents if _in_text(a, text)],
                # A deferral only makes the plan weaker, so it is kept even if unquoted.
                defers_decision=plan.defers_decision,
                defers_quote=plan.defers_quote if _in_text(plan.defers_quote, text) else None,
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
                "document": {"type": doc.type, "date": doc.date_raw, "text": doc.text},
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
