"""Generate a meaning-preserving paraphrase of the sample set.

The sample notes are templated and the heuristic extractor was calibrated on them, so
in-sample accuracy overstates robustness. This rewrites the clinically relevant phrases
(vitals, consent signatures, anticoagulation plans, medication status) into different
wording while keeping their meaning, so the human labels still apply. Scoring the
result shows how brittle an extractor is to phrasing it was not tuned on.

    uv run --with pydantic scripts/perturb.py --groups all --output data/patients_paraphrased.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]

Sub = tuple[str, str | Callable[[re.Match[str]], str]]

UNIT_WORD = {"F": "degrees Fahrenheit", "C": "degrees Celsius"}

GROUPS: dict[str, list[Sub]] = {
    "vitals": [
        (r"(?:NIBP|BP:?|Blood pressure)\s+(\d{2,3})/(\d{2,3})(?:\s*mmHg)?", r"blood pressure measured at \1 over \2"),
        (r"(?<=\d\d:\d\d  )(\d{2,3})/(\d{2,3})\s*mmHg", r"blood pressure measured at \1 over \2"),
        (r"(?:\bT|Temp:?|Temperature)\s+(\d{2,3}(?:\.\d)?)\s*°?\s*([CF])\b(?:\s*\((?:oral)\))?",
         lambda m: f"temperature of {m.group(1)} {UNIT_WORD[m.group(2)]}"),
        (r"Temp:\s*(\d{2,3}(?:\.\d)?)([CF])\b", lambda m: f"temperature of {m.group(1)} {UNIT_WORD[m.group(2)]}"),
        (r"(\d{2,3}\.\d)\s*°([CF])\s+(temporal|tympanic)",
         lambda m: f"{m.group(3)} temperature of {m.group(1)} {UNIT_WORD[m.group(2)]}"),
    ],
    "consent": [
        (r"Patient signature: pending", "The patient has not signed this form yet."),
        (r"Patient signature on file - signed in clinic (\S+) (\S+) \(scanned copy attached\)\.",
         r"Patient signed the consent in clinic on \1 at \2."),
        (r"Patient signature: /s/ ([^(]+)\(electronically signed (\S+) (\S+)\)",
         r"Patient e-signature captured on \2 at \3 (\1)."),
        (r"Signed by patient: ([^,]+), (\S+) (\S+)\.", r"\1 signed the form on \2 at \3."),
        (r"E-signature request sent to patient portal (\S+) (\S+); status: NOT COMPLETED as of this note\.",
         r"Portal e-signature request from \1 is still outstanding."),
        (r"\[Form printed for patient review; not yet signed\]",
         "Form handed to the patient to review at home; no signature obtained."),
        (r"Patient signature: _+\s+Date/Time: _+", "Signature and date fields left empty."),
    ],
    "plans": [
        (r"Hold (\w+) for the (\d+) days before surgery \(last dose ([\d/]+); no doses in the (\d+) hours prior\)\.",
         r"\1 should be paused \2 days ahead of the operation, final dose \3."),
        (r"Stop warfarin (\d+) days before surgery \(last dose ([\d/]+)\);", r"Warfarin is to be paused \1 days ahead of the operation, final dose \2;"),
        (r"Resume (\w+) ([\d-]+) hours (?:after surgery|post-op)", r"\1 may be taken again \2 hours after the operation"),
        (r"Restart (\w+) on postoperative day (\d+)", r"\1 may be taken again on postoperative day \2"),
        (r"Resume warfarin at the usual home dose the evening of surgery", "Warfarin may be taken again at the usual home dose the evening of the operation"),
        (r"Post-operative anticoagulation will be addressed after surgery\.", "We will decide about anticoagulation after the operation."),
        (r"Periprocedural management of (\w+) to be determined by the surgical team;", r"How to handle \1 around the procedure is left to the surgical team's judgment;"),
    ],
    "medications": [
        (r" - taking as prescribed", " - confirmed current"),
        (r"Patient also reports taking", "Patient mentions also using"),
        (r"patient unable to confirm whether still taking", "patient does not remember if she still uses it"),
        (r"patient states warfarin was stopped last month and replaced with Eliquis",
         "patient says cardiology swapped warfarin for Eliquis last month"),
    ],
}


def perturb_text(text: str, groups: list[str]) -> str:
    for group in groups:
        for pattern, replacement in GROUPS[group]:
            text = re.sub(pattern, replacement, text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(ROOT / "data" / "patients_sample_50.jsonl"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", default="all", help=f"comma list of {sorted(GROUPS)} or 'all'")
    args = parser.parse_args()

    groups = sorted(GROUPS) if args.groups == "all" else args.groups.split(",")
    changed = 0
    with open(args.input, encoding="utf-8") as src, open(args.output, "w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            for doc in row["submission"].get("documents", []):
                new = perturb_text(doc.get("text", ""), groups)
                changed += new != doc.get("text", "")
                doc["text"] = new
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"groups={groups} documents changed={changed} -> {args.output}")


if __name__ == "__main__":
    main()
