# Pre-Op Scheduling Triage

Evaluates one pre-op submission package against the Cadence Surgical Center scheduling
policy and returns `READY`, `NEEDS_FOLLOW_UP`, or `NOT_CLEARED`, with every issue tied to
the exact field, value, or document excerpt it came from.

**Design in one line:** an extractor (LLM or rule-based) reads each document and reports
quoted facts; deterministic code applies the policy and makes the decision. See
[WRITEUP.md](WRITEUP.md) for the approach, design decisions, assumptions, and results.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not installed
export OPENAI_API_KEY="..."                        # only for the LLM extractor / hosted evals
```

`uv` provisions Python 3.11+ and dependencies per script; there is no separate install step.

## Run

| Command | What it does | Needs API key |
|---|---|---|
| `make test` | 45 unit/edge-case tests (fake LLM client, no network) | no |
| `make baseline` | Triage all sample cases -> `data/baseline_outputs.jsonl` | only with `EXTRACTOR=llm` |
| `make evals-local` | Score outputs locally (schema, decision, categories, confusion matrix, false READYs) | no |
| `make evals` | Same scoring, plus the hosted OpenAI Evals run from the starter | yes |
| `make determinism` | Same case 10x, checks exact-output stability | only with `EXTRACTOR=llm` |
| `make robustness` | Original sample + meaning-preserving paraphrases (in- vs out-of-sample) | only with `EXTRACTOR=llm` |
| `make report` | Interactive TUI over the eval report | no |

Choose the document extractor with `EXTRACTOR=heuristic` (default) or `EXTRACTOR=llm`,
e.g. `make baseline evals-local EXTRACTOR=llm MODEL=gpt-4.1-mini`. In LLM mode the
heuristic extractor remains the per-document fallback, and responses are cached under
`data/.cache/llm/` (disable with `TRIAGE_LLM_CACHE=0`).

Programmatic use:

```python
from core import triage_submission
output = triage_submission(submission_dict, model="gpt-4.1-mini", extractor="heuristic")
print(output.model_dump_json(indent=2))
```

## Layout

```
core.py               harness entry point: triage_submission(submission, model=...)
triage/
  policy.py           all policy numbers and vocabularies (windows, thresholds, lab aliases, drug classes)
  normalize.py        lenient parse of the raw submission (never raises, never drops fields)
  schema.py           DocumentFacts: the extraction contract shared by both extractors
  heuristic.py        rule-based extractor (offline fallback, rules-only baseline)
  llm.py              OpenAI extractor: strict structured output, quote verification, cache
  tripwires.py        safety net: raw-text signals no extracted fact accounts for -> UNKNOWN
  rules.py            one function per policy rule -> PASS / FAIL / UNKNOWN findings
  engine.py           orchestration, decision precedence, deterministic rendering
scripts/perturb.py    meaning-preserving paraphrase generator
scripts/robustness.py in-sample vs paraphrase scoring table
tests/                end-to-end edge cases (one per labeled rationale pattern) + LLM extractor tests
```

## Changes to the starter harness

- `run_evals.py`: `--local-only` (score without the hosted Evals API), decision confusion
  matrix, and a `false_ready_count` in the summary.
- `run_baseline.py`: passes the **raw** submission to triage (validating through the
  starter's pydantic schema silently dropped unknown fields such as a `value_c`
  temperature, and crashed on unexpected enum values); adds `--extractor`.
- `Makefile`: `evals-local`, `robustness`, and `EXTRACTOR`.
- `tests/`: the starter tests asserted details of the single-LLM-call baseline and were
  replaced.
