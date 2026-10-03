INPUT ?= data/patients_sample_50.jsonl
OUTPUT ?= data/baseline_outputs.jsonl
REPORT ?= data/eval_report.json
DETERMINISM_REPORT ?= data/determinism_report.json
MODEL ?= gpt-4.1-mini
# auto = LLM extractor when OPENAI_API_KEY is set, rule-based extractor otherwise.
EXTRACTOR ?= auto

# Pick up OPENAI_API_KEY from a local, gitignored .env if present.
ifneq (,$(wildcard .env))
include .env
export OPENAI_API_KEY
endif

.PHONY: baseline evals evals-local determinism robustness score report test all clean

baseline:
	uv run run_baseline.py \
		--input $(INPUT) \
		--output $(OUTPUT) \
		--model $(MODEL) \
		--extractor $(EXTRACTOR)

evals:
	uv run run_evals.py \
		--input $(INPUT) \
		--outputs $(OUTPUT) \
		--report $(REPORT)

# Same scoring without the hosted OpenAI Evals run; needs no API key.
evals-local:
	uv run run_evals.py \
		--local-only \
		--input $(INPUT) \
		--outputs $(OUTPUT) \
		--report $(REPORT)

determinism:
	TRIAGE_EXTRACTOR=$(EXTRACTOR) uv run run_evals.py \
		--determinism \
		--input $(INPUT) \
		--model $(MODEL) \
		--report $(DETERMINISM_REPORT)

# Original sample plus meaning-preserving paraphrases (in-sample vs out-of-sample).
robustness:
	uv run --with 'pydantic>=2.8.0' --with 'openai>=2.0.0' \
		scripts/robustness.py --extractor $(EXTRACTOR) --model $(MODEL)

score:
	@python3 -c 'import json; r=json.load(open("$(REPORT)")); s=(r.get("primary_score",{}) or {}).get("value_pct"); print(s if s is not None else r.get("local_metrics_summary",{}).get("aggregate_local_score_pct", 0.0))'

report:
	uv run view_report.py --report $(REPORT)

test:
	uv run \
		--with 'openai>=2.0.0' \
		--with 'pydantic>=2.8.0' \
		--with 'pytest>=8.0.0' \
		python -m pytest tests

all: baseline evals determinism score

clean:
	rm -f data/baseline_outputs.jsonl \
		data/eval_report.json \
		data/determinism_report.json
