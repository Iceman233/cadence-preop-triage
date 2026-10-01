"""End-to-end policy behavior. Case numbers refer to the labeled sample cases whose
rationale motivated the test."""

from __future__ import annotations

import json

from conftest import SIGNED_CONSENT, add_doc, categories, load_fixture

from core import TriageOutput, triage_submission
from triage import engine, heuristic


def test_ready_baseline(ready, run):
    output = run(ready)
    assert isinstance(output, TriageOutput)
    assert output.decision == "READY"
    assert output.issues == []


def test_pdf_example_matches_expected_output(run):
    output = run(load_fixture("pdf_example.json"))
    assert output.decision == "NEEDS_FOLLOW_UP"
    assert [(i.category, i.evidence.source) for i in output.issues] == [
        ("MISSING_REQUIRED_DATA", "procedure.procedure_date"),
        ("ANTICOAGULATION_MANAGEMENT", "documents[4]"),
    ]
    assert output.explanation.startswith("MISSING_REQUIRED_DATA: Missing procedure date | ANTICOAGULATION_MANAGEMENT:")


def test_output_is_deterministic(ready, run):
    ready["procedure"]["procedure_date"] = None
    first, second = run(ready), run(ready)
    assert first.model_dump_json() == second.model_dump_json()


# -------------------------
# Missing data maps to MISSING_REQUIRED_DATA only
# -------------------------


def test_missing_procedure_date_reports_only_missing_data(ready, run):  # cases 11, 44
    ready["procedure"]["procedure_date"] = None
    output = run(ready)
    assert output.decision == "NEEDS_FOLLOW_UP"
    assert categories(output) == {"MISSING_REQUIRED_DATA"}
    assert "cannot evaluate: H&P recency, required test recency" in output.issues[0].evidence.details


def test_missing_risk_reports_only_missing_data(ready, run):  # cases 29, 39
    ready["procedure"]["procedure_risk"] = None
    assert categories(run(ready)) == {"MISSING_REQUIRED_DATA"}


def test_unrecognized_risk_does_not_crash(ready, run):
    ready["procedure"]["procedure_risk"] = "MEDIUM"
    output = run(ready)
    assert categories(output) == {"MISSING_REQUIRED_DATA"}
    assert "MEDIUM" in output.issues[0].evidence.details


def test_risk_casing_is_tolerated(ready, run):
    ready["procedure"]["procedure_risk"] = "low"
    assert run(ready).decision == "READY"


def test_missing_temperature_is_missing_data(ready, run):  # cases 9, 21
    ready["vitals"] = [v for v in ready["vitals"] if v["type"] != "temperature"]
    add_doc(ready, "Nursing Pre-procedure Assessment", "2026-03-18", "VITAL SIGNS: See flowsheet.")
    assert categories(run(ready)) == {"MISSING_REQUIRED_DATA"}


# -------------------------
# Rule 1: H&P and consent
# -------------------------


def test_hp_exactly_30_days_passes_31_fails(ready, run):  # cases 12, 14
    ready["documents"][0]["date"] = "2026-02-18"  # 30 days before 03-20
    assert run(ready).decision == "READY"
    ready["documents"][0]["date"] = "2026-02-17"  # 31 days
    assert categories(run(ready)) == {"REQUIRED_DOCUMENTATION"}


def test_hp_found_by_content_despite_typo_title(ready, run):  # cases 1, 45
    ready["documents"][0]["type"] = "History and Pyhsical"
    assert run(ready).decision == "READY"


def test_missing_hp(ready, run):  # cases 13, 48
    del ready["documents"][0]
    output = run(ready)
    assert categories(output) == {"REQUIRED_DOCUMENTATION"}
    assert output.issues[0].description == "Missing History and Physical (H&P)"


def test_physician_attestation_is_not_patient_signature(ready, run):  # case 49 trap
    ready["documents"][1]["text"] = SIGNED_CONSENT.replace(
        "Patient signature on file - signed in clinic 03/05/2026 09:00 (scanned copy attached).",
        "Patient signature: pending",
    )
    output = run(ready)
    assert categories(output) == {"REQUIRED_DOCUMENTATION"}
    assert "PENDING" in output.issues[0].evidence.details


def test_portal_consent_not_completed(ready, run):  # cases 2, 28
    ready["documents"][1]["text"] = SIGNED_CONSENT.replace(
        "Patient signature on file - signed in clinic 03/05/2026 09:00 (scanned copy attached).",
        "E-signature request sent to patient portal 03/05/2026 11:22; status: NOT COMPLETED as of this note.",
    )
    assert categories(run(ready)) == {"REQUIRED_DOCUMENTATION"}


def test_newer_signed_consent_supersedes_pending(ready, run):  # cases 7, 49
    ready["documents"][1]["date"] = "2026-03-12"
    add_doc(ready, "Procedure Consent Form", "2026-03-02", SIGNED_CONSENT.replace(
        "Patient signature on file - signed in clinic 03/05/2026 09:00 (scanned copy attached).",
        "Patient signature: pending",
    ))
    assert run(ready).decision == "READY"


def test_consent_for_different_procedure(ready, run):  # case 24
    ready["documents"][1]["text"] = SIGNED_CONSENT.replace("carpal tunnel release", "coronary artery bypass grafting")
    output = run(ready)
    assert categories(output) == {"REQUIRED_DOCUMENTATION"}
    assert output.issues[0].description == "Surgical consent is for a different procedure"


# -------------------------
# Rule 2: testing
# -------------------------


def test_entered_in_error_lab_is_skipped(ready, run):  # cases 4, 6, 41
    ready["labs"][0]["effective_at"] = "2026-01-10T08:00:00Z"  # 69 days: outside window
    ready["labs"].append({"code": "CBC", "effective_at": "2026-03-15T08:00:00Z", "status": "entered-in-error"})
    output = run(ready)
    assert categories(output) == {"REQUIRED_TESTING"}
    assert "entered-in-error" in output.issues[0].evidence.details


def test_lab_code_aliases(ready, run):
    ready["labs"][0].update(code="LAB-CBC", display="Complete Blood Count w/ Differential")
    assert run(ready).decision == "READY"


def test_high_risk_needs_cmp_and_bmp_does_not_count(ready, run):  # case 17
    ready["procedure"]["procedure_risk"] = "HIGH"
    ready["labs"].append({"code": "BMP", "display": "Basic Metabolic Panel", "effective_at": "2026-03-12T08:00:00Z", "status": "final"})
    ready["labs"][0]["effective_at"] = "2026-03-10T08:00:00Z"
    output = run(ready)
    assert [i.description for i in output.issues] == ["Missing required CMP (HIGH risk)"]


def test_high_risk_15_days_fails(ready, run):  # case 9
    ready["procedure"]["procedure_risk"] = "HIGH"
    ready["labs"] = [
        {"code": "CBC", "effective_at": "2026-03-05T08:00:00Z", "status": "final"},  # 15 days
        {"code": "CMP", "effective_at": "2026-03-06T08:00:00Z", "status": "final"},  # 14 days
    ]
    assert [i.description for i in run(ready).issues] == ["CBC outside 14-day window"]


# -------------------------
# Rule 3: anticoagulation
# -------------------------

COMPLETE_PLAN = (
    "RECOMMENDATIONS:\n"
    "  1. Hold Xarelto for the 2 days before surgery (last dose 03/17/2026). "
    "Resume Xarelto 24-48 hours after surgery once hemostasis is secure."
)


def test_antiplatelets_need_no_plan(ready, run):  # cases 2, 25
    ready["medications"] = [
        {"name": "ASA 81 mg PO daily", "active": True},
        {"name": "clopidogrel 75 mg PO daily", "active": True},
    ]
    assert run(ready).decision == "READY"


def test_brand_name_anticoagulant_with_complete_plan(ready, run):  # case 37
    ready["medications"] = [{"name": "Xarelto 20 mg PO daily", "active": True}]
    add_doc(ready, "Perioperative Medication Plan", "2026-03-10", COMPLETE_PLAN)
    assert run(ready).decision == "READY"


def test_anticoagulant_without_plan(ready, run):
    ready["medications"] = [{"name": "Eliquis 5 mg PO BID", "active": True}]
    output = run(ready)
    assert categories(output) == {"ANTICOAGULATION_MANAGEMENT"}
    assert output.issues[0].evidence.source == "medications[0]"


def test_plan_missing_post_op_restart(ready, run):  # case 20
    ready["medications"] = [{"name": "warfarin 5 mg PO daily", "active": True}]
    add_doc(ready, "Cardiology Progress Note", "2026-03-10",
            "RECOMMENDATIONS:\n  1. Stop warfarin 5 days before surgery (last dose 03/15/2026). "
            "Bridge with enoxaparin 1 mg/kg SC q12h, last dose 24 hours pre-op. "
            "Post-operative anticoagulation will be addressed after surgery.")
    output = run(ready)
    assert categories(output) == {"ANTICOAGULATION_MANAGEMENT"}
    assert "post-operative management not specified" in output.issues[0].description


def test_plan_deferred_to_surgical_team(ready, run):  # cases 23, 36
    ready["medications"] = [{"name": "dabigatran 150 mg PO BID", "active": True}]
    add_doc(ready, "Cardiology Progress Note", "2026-03-10",
            "1. Periprocedural management of dabigatran to be determined by the surgical team; happy to assist.")
    assert categories(run(ready)) == {"ANTICOAGULATION_MANAGEMENT"}


def test_plan_for_previous_drug_does_not_cover_new_one(ready, run):  # case 18
    ready["medications"] = [
        {"name": "apixaban 5 mg PO BID", "active": True},
        {"name": "warfarin 5 mg PO daily", "active": False},
    ]
    add_doc(ready, "Perioperative Medication Plan", "2026-02-20",
            "1. Stop warfarin 5 days before surgery (last dose 03/15/2026). "
            "Resume warfarin the evening of surgery.")
    add_doc(ready, "Nursing Pre-procedure Assessment", "2026-03-17",
            "- Note: patient states warfarin was stopped last month and replaced with Eliquis by cardiology.")
    output = run(ready)
    assert categories(output) == {"ANTICOAGULATION_MANAGEMENT"}
    assert "apixaban" in output.issues[0].description


def test_anticoagulant_reported_only_in_notes(ready, run):  # cases 0, 27, 38
    add_doc(ready, "Pre-Admission Testing Visit", "2026-03-15",
            "- Patient also reports taking Eliquis 5 mg (BID), prescribed by an outside cardiologist.")
    assert categories(run(ready)) == {"ANTICOAGULATION_MANAGEMENT"}


def test_inactive_anticoagulant_needs_no_plan(ready, run):  # cases 19, 43
    ready["medications"] = [{"name": "warfarin (Coumadin) 5 mg PO daily", "active": False}]
    add_doc(ready, "H&P addendum", "2026-03-02",
            "Completed 6 months of warfarin, discontinued by hematology in early 2020. Not currently anticoagulated.")
    assert run(ready).decision == "READY"


def test_unconfirmed_anticoagulant_is_missing_data(ready, run):  # case 33
    ready["medications"] = [{"name": "Coumadin 5 mg PO daily", "active": None}]
    add_doc(ready, "Pre-Admission Testing Visit", "2026-03-15",
            "- Coumadin: patient unable to confirm whether still taking; pharmacy fill history requested")
    assert categories(run(ready)) == {"MISSING_REQUIRED_DATA"}


# -------------------------
# Rule 4: acute safety
# -------------------------


def test_newer_note_vitals_override_structured(ready, run):  # cases 5, 47
    add_doc(ready, "Nursing Pre-procedure Assessment", "2026-03-18",
            "VITAL SIGNS:\n  Vitals 09:10  BP 157/117 mmHg, HR 80, T 98.6 F, SpO2 98% RA")
    output = run(ready)
    assert output.decision == "NOT_CLEARED"
    assert output.issues[0].evidence.source == "documents[2]"


def test_recheck_with_trailing_colon_time_wins(ready, run):  # cases 16, 32
    add_doc(ready, "Pre-Admission Testing Visit", "2026-03-18",
            "VITAL SIGNS: On arrival at 08:06: BP 189/112 mmHg, T 98.6 F. "
            "After resting quietly for 24 minutes, BP was rechecked at 08:30: BP 137/78 mmHg.")
    assert run(ready).decision == "READY"


def test_celsius_conversion(ready, run):  # cases 30, 35
    add_doc(ready, "Pre-op Nursing Intake", "2026-03-18", "Vitals 09:00  BP 120/70, Temp: 37.9C")
    assert run(ready).decision == "READY"  # 100.2 F
    ready["documents"][-1]["text"] = "Vitals 09:00  BP 120/70, Temp: 38.6C"
    assert run(ready).decision == "NOT_CLEARED"  # 101.5 F


def test_diastolic_exactly_110_is_not_cleared(ready, run):  # case 28
    ready["vitals"][0]["diastolic"] = 110
    assert run(ready).decision == "NOT_CLEARED"


def test_structured_celsius_field_is_not_dropped(ready, run):
    ready["vitals"].append({"type": "temperature", "value_c": 38.6, "date": "2026-03-19T09:00:00Z"})
    assert run(ready).decision == "NOT_CLEARED"


def test_not_cleared_still_lists_other_issues(ready, run):  # cases 2, 28, 31
    ready["vitals"][0]["systolic"] = 185
    del ready["documents"][1]
    output = run(ready)
    assert output.decision == "NOT_CLEARED"
    assert categories(output) == {"ACUTE_SAFETY_EXCLUSION", "REQUIRED_DOCUMENTATION"}


# -------------------------
# Fail-safe behavior
# -------------------------


def test_tripwire_catches_vital_missed_by_extractor(ready):
    """An extractor that silently drops vitals must not yield READY."""

    def lossy(doc, context):
        facts = heuristic.extract(doc, context)
        return facts.model_copy(update={"vitals": []})

    add_doc(ready, "Nursing Pre-procedure Assessment", "2026-03-18", "Vitals 09:10  BP 157/117 mmHg, T 98.6 F")
    output = engine.triage(ready, [("lossy", lossy)])
    assert output.decision == "NEEDS_FOLLOW_UP"
    assert {i.description for i in output.issues} == {
        "Unverified blood pressure in documents[2]",
        "Unverified temperature in documents[2]",
    }


def test_tripwire_is_broader_than_extractor(ready, run):
    """Phrasing the heuristic cannot parse must still block READY (paraphrase suite finding)."""

    add_doc(ready, "Nursing Pre-procedure Assessment", "2026-03-18",
            "VITAL SIGNS: blood pressure measured at 157 over 117, temperature of 101.2 degrees Fahrenheit")
    output = run(ready)
    assert output.decision == "NEEDS_FOLLOW_UP"
    assert {i.description for i in output.issues} == {
        "Unverified blood pressure in documents[2]",
        "Unverified temperature in documents[2]",
    }


def test_extractor_failure_falls_back_then_fails_closed(ready):
    def broken(doc, context):
        raise RuntimeError("model unavailable")

    assert engine.triage(ready, [("broken", broken), ("heuristic", heuristic.extract)]).decision == "READY"
    output = engine.triage(ready, [("broken", broken)])
    assert output.decision == "NEEDS_FOLLOW_UP"
    assert categories(output) >= {"MISSING_REQUIRED_DATA"}


def test_output_schema_is_valid_json(ready):
    output = triage_submission(ready, model="unused")
    assert TriageOutput.model_validate(json.loads(output.model_dump_json())) == output
