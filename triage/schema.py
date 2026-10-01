"""Facts extracted from one clinical document.

This is the contract between extraction (heuristic or LLM) and the rules. It records
what a document *says*, each claim backed by a verbatim quote; it never records a
policy conclusion. Cross-document reasoning (supersession, drug matching, recency)
happens in code, in rules.py.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DocKind = Literal["HISTORY_AND_PHYSICAL", "SURGICAL_CONSENT", "OTHER"]
Signature = Literal["SIGNED", "PENDING", "NOT_COMPLETED", "ABSENT", "UNCLEAR"]
StepStatus = Literal["SPECIFIC", "VAGUE", "ABSENT"]


class HPFacts(BaseModel):
    purpose: Literal["PREOP_EVALUATION", "NON_PREOP", "UNCLEAR"] = Field(
        description="PREOP_EVALUATION for a pre-operative workup; NON_PREOP for wellness, "
        "establish-care, or a copy retained for chart context."
    )
    purpose_quote: str | None


class ConsentFacts(BaseModel):
    procedure_named: str | None = Field(description="Procedure the consent authorizes, as written.")
    procedure_quote: str | None
    patient_signature: Signature = Field(
        description="State of the PATIENT's signature only. A physician attestation or "
        "physician e-signature is never the patient's signature."
    )
    patient_signature_quote: str | None
    patient_signed_date: str | None = Field(description="Date the patient signed, as written.")


class VitalReading(BaseModel):
    kind: Literal["BLOOD_PRESSURE", "TEMPERATURE"]
    systolic: float | None
    diastolic: float | None
    temp_value: float | None
    temp_unit: Literal["F", "C"] | None
    time_of_day: str | None = Field(description="HH:MM if the reading has a time.")
    quote: str = Field(description="Verbatim text containing the numbers.")


class MedicationMention(BaseModel):
    name_as_written: str
    status: Literal["TAKING", "STOPPED", "UNCERTAIN"]
    llm_drug_class: Literal["ANTICOAGULANT", "ANTIPLATELET", "OTHER", "UNKNOWN"]
    quote: str


class PlanStep(BaseModel):
    status: StepStatus = Field(
        description="SPECIFIC: a concrete action with timing (e.g. 'stop 5 days before', "
        "'restart on POD 1'). VAGUE: mentioned without a concrete action "
        "('will be addressed after surgery'). ABSENT: not mentioned."
    )
    quote: str | None


class AnticoagPlan(BaseModel):
    drug_as_written: str | None = Field(description="Anticoagulant the plan manages; null if unnamed.")
    pre_op: PlanStep
    post_op: PlanStep
    bridging_agents: list[str]
    defers_decision: bool = Field(
        description="True if management is handed off ('to be determined by the surgical "
        "team', 'recommendations to follow')."
    )
    defers_quote: str | None


class DocumentFacts(BaseModel):
    doc_kind: DocKind
    hp: HPFacts | None
    consent: ConsentFacts | None
    vitals: list[VitalReading]
    medication_mentions: list[MedicationMention]
    anticoag_plans: list[AnticoagPlan]

    @classmethod
    def empty(cls) -> DocumentFacts:
        return cls(
            doc_kind="OTHER",
            hp=None,
            consent=None,
            vitals=[],
            medication_mentions=[],
            anticoag_plans=[],
        )
