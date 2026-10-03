"""Score an extractor on the original sample and on each paraphrase group.

Same metrics as run_evals.py (schema, decision, category set) plus false READYs, run
in-process. This is the in-sample vs out-of-sample table in the write-up.

    uv run --with pydantic --with openai scripts/robustness.py --extractor heuristic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from perturb import GROUPS, perturb_text  # noqa: E402

from core import resolve_mode, triage_submission  # noqa: E402


def score(rows: list[dict], extractor: str, model: str) -> dict[str, float]:
    totals = {"decision": 0, "categories": 0, "false_ready": 0}
    for row in rows:
        output = triage_submission(row["submission"], model=model, extractor=extractor)
        label = row["label"]
        totals["decision"] += output.decision == label["decision"]
        totals["categories"] += sorted({i.category for i in output.issues}) == sorted(set(label["categories"]))
        totals["false_ready"] += output.decision == "READY" and label["decision"] != "READY"
    n = len(rows)
    return {
        # Schema validity is guaranteed by the pydantic output model, so it counts as 1.
        "score": round(100 * (1 + totals["decision"] / n + totals["categories"] / n) / 3, 1),
        "decision": round(100 * totals["decision"] / n, 1),
        "categories": round(100 * totals["categories"] / n, 1),
        "false_ready": totals["false_ready"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(ROOT / "data" / "patients_sample_50.jsonl"))
    parser.add_argument("--extractor", choices=["auto", "heuristic", "llm"], default="auto")
    parser.add_argument("--model", default="gpt-4.1-mini")
    args = parser.parse_args()

    base = [json.loads(line) for line in open(args.input, encoding="utf-8")]
    sets = {"original": []} | {g: [g] for g in sorted(GROUPS)} | {"all": sorted(GROUPS)}

    print(f"extractor={resolve_mode(args.extractor)}")
    print(f"{'set':<12} {'score':>6} {'decision':>9} {'categories':>11} {'false_ready':>12}")
    for name, groups in sets.items():
        rows = json.loads(json.dumps(base))
        for row in rows:
            for doc in row["submission"].get("documents", []):
                doc["text"] = perturb_text(doc.get("text", ""), groups)
        s = score(rows, args.extractor, args.model)
        print(f"{name:<12} {s['score']:>6} {s['decision']:>8}% {s['categories']:>10}% {s['false_ready']:>12}")


if __name__ == "__main__":
    main()
