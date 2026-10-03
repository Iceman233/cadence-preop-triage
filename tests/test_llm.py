"""LLM extractor behavior with a fake client (no network, no API key)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from conftest import add_doc, categories

from triage import engine, heuristic
from triage.llm import LLMExtractor
from triage.normalize import Document
from triage.schema import ConsentFacts, DocumentFacts, VitalReading


def fake_client(*facts: DocumentFacts) -> Mock:
    client = Mock()
    client.responses.parse.side_effect = [SimpleNamespace(output_parsed=f) for f in facts]
    return client


def doc(text: str, type_: str = "Nursing Pre-procedure Assessment") -> Document:
    return Document(index=0, type=type_, date=None, date_raw="2026-03-18", text=text)


def bp(systolic: float, diastolic: float, quote: str) -> VitalReading:
    return VitalReading(kind="BLOOD_PRESSURE", systolic=systolic, diastolic=diastolic, temp_value=None,
                        temp_unit=None, time_of_day=None, quote=quote)


def test_request_uses_strict_schema_and_temperature_zero(tmp_path):
    client = fake_client(DocumentFacts.empty())
    LLMExtractor(model="gpt-test", client=client, cache_dir=tmp_path)(doc("No vitals."), {})
    kwargs = client.responses.parse.call_args.kwargs
    assert kwargs["model"] == "gpt-test"
    assert kwargs["text_format"] is DocumentFacts
    assert kwargs["temperature"] == 0


def test_fabricated_vital_quote_is_dropped(tmp_path):
    text = "Vitals 09:10  BP 157/117 mmHg"
    hallucinated = DocumentFacts.empty().model_copy(update={"vitals": [bp(120, 80, "BP 120/80 mmHg")]})
    facts = LLMExtractor(model="m", client=fake_client(hallucinated), cache_dir=tmp_path)(doc(text), {})
    assert facts.vitals == []


def test_vital_numbers_must_appear_in_quote(tmp_path):
    text = "Vitals 09:10  BP 157/117 mmHg"
    misread = DocumentFacts.empty().model_copy(update={"vitals": [bp(137, 78, "BP 157/117 mmHg")]})
    facts = LLMExtractor(model="m", client=fake_client(misread), cache_dir=tmp_path)(doc(text), {})
    assert facts.vitals == []


def test_model_sees_ascii_and_evidence_keeps_original_wording(tmp_path):
    """Observed live: copying '°C' produced '\\x00b0C' / '\\x176'. The model gets ASCII text,
    and its quote is mapped back to the document's exact span for evidence."""

    text = "08:44  BP 127/80, Temperature 36.6 °C (oral), HR 82"
    reading = VitalReading(kind="TEMPERATURE", systolic=None, diastolic=None, temp_value=36.6, temp_unit="C",
                           time_of_day="08:44", quote="Temperature 36.6  C (oral)")
    client = fake_client(DocumentFacts.empty().model_copy(update={"vitals": [reading]}))
    facts = LLMExtractor(model="m", client=client, cache_dir=tmp_path)(doc(text), {})
    assert "°" not in client.responses.parse.call_args.kwargs["input"][0]["content"]
    assert [(v.temp_value, v.quote) for v in facts.vitals] == [(36.6, "Temperature 36.6 °C (oral)")]


def test_garbled_quote_is_rejected(tmp_path):
    text = "08:44  BP 127/80, Temperature 36.6 °C (oral), HR 82"
    reading = VitalReading(kind="TEMPERATURE", systolic=None, diastolic=None, temp_value=36.6, temp_unit="C",
                           time_of_day="08:44", quote="Temperature 36.6 \x176 (oral)")
    facts = LLMExtractor(model="m", client=fake_client(DocumentFacts.empty().model_copy(update={"vitals": [reading]})),
                         cache_dir=tmp_path)(doc(text), {})
    assert facts.vitals == []


def test_combined_bp_and_temperature_entry_is_split(tmp_path):
    """Observed live: one entry carried both a BP and a temperature."""

    text = "08:44  BP 127/80, Temperature 36.6 °C (oral)"
    combined = VitalReading(kind="BLOOD_PRESSURE", systolic=127, diastolic=80, temp_value=36.6, temp_unit="C",
                            time_of_day="08:44", quote=text)
    facts = LLMExtractor(model="m", client=fake_client(DocumentFacts.empty().model_copy(update={"vitals": [combined]})),
                         cache_dir=tmp_path)(doc(text), {})
    assert sorted(v.kind for v in facts.vitals) == ["BLOOD_PRESSURE", "TEMPERATURE"]


def test_unquoted_signature_is_downgraded(tmp_path):
    text = "INFORMED CONSENT\nProcedure: carpal tunnel release\nPatient signature: pending"
    claimed = DocumentFacts.empty().model_copy(update={
        "doc_kind": "SURGICAL_CONSENT",
        "consent": ConsentFacts(procedure_named="carpal tunnel release", procedure_quote="Procedure: carpal tunnel release",
                                patient_signature="SIGNED", patient_signature_quote="Signed by patient",
                                patient_signed_date="03/05/2026"),
    })
    facts = LLMExtractor(model="m", client=fake_client(claimed), cache_dir=tmp_path)(doc(text, "Consent"), {})
    assert facts.consent.patient_signature == "UNCLEAR"


def test_cache_avoids_repeat_calls(tmp_path):
    client = fake_client(DocumentFacts.empty())
    extractor = LLMExtractor(model="m", client=client, cache_dir=tmp_path)
    extractor(doc("No vitals."), {})
    extractor(doc("No vitals."), {})
    assert client.responses.parse.call_count == 1


def test_document_identity_comes_from_title_not_model(ready):
    """Observed live: the model labeled a pre-admission visit an H&P, which would
    falsely satisfy Rule 1. Identity is fixed from title/heading."""

    del ready["documents"][0]  # remove the real H&P
    add_doc(ready, "Pre-Admission Testing Visit", "2026-03-12",
            "PRE-ADMISSION TESTING VISIT\nPlanned procedure: carpal tunnel release\nHistory reviewed; exam unremarkable.")

    def over_inclusive(document, context):
        facts = heuristic.extract(document, context)
        if "Pre-Admission" in document.type:
            facts = facts.model_copy(update={"doc_kind": "HISTORY_AND_PHYSICAL"})
        return facts

    output = engine.triage(ready, [("llm", over_inclusive)])
    assert [i.description for i in output.issues] == ["Missing History and Physical (H&P)"]


def test_llm_miss_is_caught_by_tripwire(ready):
    """The model omits a hypertensive reading: tripwire turns READY into follow-up."""

    add_doc(ready, "Nursing Pre-procedure Assessment", "2026-03-18", "Vitals 09:10  BP 157/117 mmHg, T 98.6 F")

    def llm_like(document, context):
        facts = heuristic.extract(document, context)
        return facts.model_copy(update={"vitals": [v for v in facts.vitals if v.kind != "BLOOD_PRESSURE"]})

    output = engine.triage(ready, [("llm", llm_like)])
    assert output.decision == "NEEDS_FOLLOW_UP"
    assert categories(output) == {"MISSING_REQUIRED_DATA"}


def test_llm_errors_fall_back_to_heuristic(ready, tmp_path):
    client = Mock()
    client.responses.parse.side_effect = TimeoutError("upstream timeout")
    extractor = LLMExtractor(model="m", client=client, cache_dir=tmp_path)
    output = engine.triage(ready, [("llm", extractor), ("heuristic", heuristic.extract)])
    assert output.decision == "READY"
