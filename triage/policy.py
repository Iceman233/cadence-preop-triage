"""Cadence pre-op scheduling policy (effective 2026-01-01) as data.

Every number and lookup list the rules depend on lives here, so a policy change is a
config change. Drug and lab vocabularies are record-interpretation aids (brand names,
lab code aliases), not added clinical criteria.
"""

from __future__ import annotations

import re

# Rule 1: H&P must be completed within this many days of the procedure (inclusive).
H_AND_P_WINDOW_DAYS = 30

# Rule 2: required tests and their windows (days before procedure, inclusive) by risk.
REQUIRED_TESTS: dict[str, dict[str, int]] = {
    "LOW": {"CBC": 30},
    "MODERATE": {"CBC": 30},
    "HIGH": {"CBC": 14, "CMP": 14},
}

# Lab results with these statuses count; anything else (entered-in-error, preliminary,
# cancelled, ...) is not a usable result. A missing status is treated as usable.
USABLE_LAB_STATUSES = {"final", "amended", "corrected"}

# Canonical test -> (codes, display patterns). A BMP is deliberately not a CMP.
LAB_ALIASES: dict[str, tuple[set[str], re.Pattern[str]]] = {
    "CBC": (
        {"CBC", "LAB-CBC", "CBC W/ DIFF", "CBCD"},
        re.compile(r"complete blood count|\bcbc\b|hemogram", re.I),
    ),
    "CMP": (
        {"CMP", "LAB-CMP"},
        re.compile(r"comprehensive metabolic panel|\bcmp\b", re.I),
    ),
}

# Rule 4: acute safety exclusions.
SYSTOLIC_EXCLUSION_MMHG = 180  # >=
DIASTOLIC_EXCLUSION_MMHG = 110  # >=
TEMPERATURE_EXCLUSION_F = 100.4  # >

# A temperature written without a unit is read as Celsius below this value.
CELSIUS_INFERENCE_CEILING = 50.0

ANTICOAGULANT = "ANTICOAGULANT"
ANTIPLATELET = "ANTIPLATELET"

# generic name -> (class, names it may be written as). Antiplatelets are listed so they
# are positively recognized as "not an anticoagulant" rather than silently unknown.
DRUGS: dict[str, tuple[str, tuple[str, ...]]] = {
    "warfarin": (ANTICOAGULANT, ("warfarin", "coumadin", "jantoven")),
    "apixaban": (ANTICOAGULANT, ("apixaban", "eliquis")),
    "rivaroxaban": (ANTICOAGULANT, ("rivaroxaban", "xarelto")),
    "dabigatran": (ANTICOAGULANT, ("dabigatran", "pradaxa")),
    "edoxaban": (ANTICOAGULANT, ("edoxaban", "savaysa", "lixiana")),
    "betrixaban": (ANTICOAGULANT, ("betrixaban", "bevyxxa")),
    "enoxaparin": (ANTICOAGULANT, ("enoxaparin", "lovenox")),
    "dalteparin": (ANTICOAGULANT, ("dalteparin", "fragmin")),
    "heparin": (ANTICOAGULANT, ("heparin",)),
    "fondaparinux": (ANTICOAGULANT, ("fondaparinux", "arixtra")),
    "aspirin": (ANTIPLATELET, ("aspirin", "asa", "ecotrin")),
    "clopidogrel": (ANTIPLATELET, ("clopidogrel", "plavix")),
    "prasugrel": (ANTIPLATELET, ("prasugrel", "effient")),
    "ticagrelor": (ANTIPLATELET, ("ticagrelor", "brilinta")),
}

# Agents used to bridge around an interrupted anticoagulant; they appear inside plans
# for another drug and do not need a plan of their own.
BRIDGING_AGENTS = {"enoxaparin", "dalteparin", "heparin", "fondaparinux"}

_NAME_TO_GENERIC = {
    alias: generic for generic, (_, aliases) in DRUGS.items() for alias in aliases
}
_DRUG_NAME_RE = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, _NAME_TO_GENERIC), key=len, reverse=True)) + r")\b",
    re.I,
)


def find_drugs(text: str, *, drug_class: str | None = None) -> list[tuple[str, re.Match[str]]]:
    """Return (generic, match) for each recognized drug name in text, in order."""

    found = []
    for match in _DRUG_NAME_RE.finditer(text or ""):
        alias = match.group(1).lower()
        # "ASA" is also the anesthesia physical-status score ("ASA physical status: II").
        if alias == "asa" and re.match(r"\s*(physical|class|status|[IV]+\b)", text[match.end():]):
            continue
        generic = _NAME_TO_GENERIC[alias]
        if drug_class is None or DRUGS[generic][0] == drug_class:
            found.append((generic, match))
    return found


def classify_drug(name: str) -> tuple[str | None, str | None]:
    """Return (generic, class) for the first recognized drug in a medication name."""

    hits = find_drugs(name)
    if not hits:
        return None, None
    generic = hits[0][0]
    return generic, DRUGS[generic][0]
