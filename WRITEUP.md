# Write-up: Pre-Op Scheduling Triage

## Approach

**The LLM reads; code decides.** The baseline asks one model call to do everything at
once: date arithmetic, lab-code reconciliation, brand-name lookup, precedence, and
evidence. Those are the things LLMs do least reliably, and none of the result can be
audited. I split the problem along that line:

```
raw JSON ─► lenient parse ─► per-document extraction ─► tripwires ─► rule checks ─► decision + rendering
            (never lossy)     LLM or heuristic,          raw-text     PASS/FAIL/      precedence, stable
                              quoted facts only          cross-check  UNKNOWN         sort, template text
```

- **Extraction** (`schema.py`) returns what a document *says*: document kind, consent
  signature state, vital readings, medication mentions, anticoagulation plan steps. Each
  claim carries a verbatim quote. It never returns a policy conclusion.
- **Rules** (`rules.py`) are plain functions, one per policy rule, so every decision is
  traceable to a unit-tested line of code. Cross-document reasoning lives here
  (supersession of consents, matching a plan to the *current* drug, recency).
- **Rendering** is deterministic, so identical facts give byte-identical output.

## Key decisions

1. **One principle for categories instead of special cases.** A rule that FAILS emits its
   own category; a rule that is UNKNOWN because an input is absent emits
   `MISSING_REQUIRED_DATA`. A document that is present but ambiguous is a FAIL (the policy
   says "incomplete or ambiguous" means follow-up under that rule). This one principle
   reproduces every labeled case, including null procedure date → only
   `MISSING_REQUIRED_DATA` (window checks are blocked, not failed) and a missing H&P →
   `REQUIRED_DOCUMENTATION`. Checks blocked by a missing field fold into that field's
   issue ("cannot evaluate: H&P recency, required test recency").
2. **Fail closed, measured.** A wrong READY costs far more than an extra follow-up, so
   I accept some accuracy loss to avoid it. Unrecognized wording becomes UNCLEAR, which
   is a FAIL; extractor errors fall back to the heuristic, then to UNKNOWN; the eval
   reports a confusion matrix and a `false_ready_count`.
3. **Tripwires.** Errors that raise are easy to handle. The dangerous ones are silent:
   an extractor that simply misses "BP 157/117". Every document is also scanned with
   deliberately broad patterns (BP-like pairs, temperature-like values, anticoagulant
   names, unsigned-consent wording). Any signal no extracted fact accounts for becomes
   UNKNOWN, so a miss can cause follow-up but never READY.
4. **One extraction call per document, not per case.** This keeps judgment in testable
   code (the model cannot "decide" a consent is superseded), contains failures to one
   document, scales to real charts, and caches by document hash.
5. **Quote verification.** LLM claims are checked in code: quotes must appear in the
   text, and vital values must appear inside their own quote. A claim that would help a
   case pass but cannot be verified is dropped or downgraded (SIGNED → UNCLEAR,
   SPECIFIC → ABSENT). Separate quotes per claim (pre-op step vs post-op step) prevent
   one real quote from vouching for an invented claim.
6. **The raw input is the source of truth.** The starter's input schema dropped unknown
   fields (a `value_c` fever became an empty reading) and crashed on `"Moderate"`. Parsing
   is lenient and index-preserving, so evidence cites `labs[3]` exactly.
7. **Policy as data.** Windows, thresholds, lab aliases, and drug classes live in
   `policy.py`. Lookup lists are checked before any model judgment: aspirin and
   clopidogrel are antiplatelets regardless of what a model says; the LLM's drug class is
   used only for names the list does not know.

## Results

Sample set (50 cases), `make evals-local` / `make robustness`:

| Set | Heuristic score | Heuristic decision | False READY | LLM extractor |
|---|---|---|---|---|
| Original (in-sample) | 100.0 | 100% | 0 | *pending API key* |
| Paraphrased vitals | 57.3 | 54% | 0 | *pending* |
| Paraphrased consent | 63.3 | 66% | 0 | *pending* |
| Paraphrased plans | 93.3 | 92% | 0 | *pending* |
| Paraphrased medications | 99.3 | 100% | 0 | *pending* |
| All paraphrases | 53.3 | 54% | 0 | *pending* |

Determinism: 100% exact-output match over 10 runs (heuristic extractor).

**How to read this.** The 100% is in-sample: the notes are templated and I calibrated
the heuristic on these 50 cases, so it overstates generalization. To get an honest
signal I wrote `scripts/perturb.py`, which rewrites the decision-relevant phrases with
the same meaning (labels still apply). The heuristic's accuracy drops sharply, as
expected; that gap is what the LLM extractor is for. More importantly, the paraphrase
suite caught a real design flaw: the first tripwire reused the extractor's own regex,
so "blood pressure measured at 157 over 117" slipped past both, giving **4 false
READYs**. A safety net with the extractor's blind spots is not a safety net. After
making the tripwire independently broader, false READYs are 0 on every set, and
reworded vitals now land in `NEEDS_FOLLOW_UP` as "unverified." I deliberately did not
tune the extractor to the paraphrases, which would just overfit a second dataset.

Spot-checking evidence also caught a bug the score could not: case 32 was correctly
NOT_CLEARED but cited the arrival BP (196/110) instead of the later recheck (181/98),
because times written as "at 09:38:" were not parsed. Fixed and covered by a test.

## Assumptions

| Area | Assumption |
|---|---|
| Review time | `metadata.submission_received_at`; readings dated after it are ignored. |
| Windows | Calendar days, inclusive (exactly 30 passes). Labs/H&Ps dated after the procedure do not count. |
| H&P | Any H&P within 30 days counts (the policy does not require a "pre-op" H&P; no non-pre-op H&P in the data falls in the window). Identified by content, not title. If the in-text date of service is older than the document date, the older date is used. |
| Consent | Only the **patient's** signature counts, never the physician attestation. A newer consent for the same procedure supersedes an older one. Only an actual procedure mismatch fails; a consent naming no procedure is accepted (as in the prompt's example). |
| Labs | Usable statuses: final, amended, corrected, or missing. Matched by code or display (CBC/LAB-CBC/Hemogram). A BMP is not a CMP. |
| Anticoagulants | Active if listed `active: true` or a note says the patient takes it, unless a later fact says stopped. `active: null` or "unable to confirm" → `MISSING_REQUIRED_DATA`. A plan must name the same drug (brand → generic) and give a concrete pre-op **and** post-op step. Bridging agents (enoxaparin) need no plan of their own. Antiplatelets never need one. |
| Vitals | BP and temperature evaluated independently. Order: date, then time of day; a same-day note reading without a time sorts after timed readings; on an exact tie the more severe reading wins. Celsius converted; a unitless value under 50 is read as Celsius. No BP or no temperature → `MISSING_REQUIRED_DATA`. |
| Precedence | NOT_CLEARED > NEEDS_FOLLOW_UP > READY; all issues are listed regardless. |

## Limitations and next steps

- **LLM results pending.** The extractor, verification, cache, and fallback are built
  and tested with a fake client, and the schema is verified against OpenAI's strict
  format. Still to do: run `make robustness EXTRACTOR=llm` and fill in the table.
- **Unlisted drugs on the structured med list** are not classified (the LLM classifies
  unknown names only inside documents). A small classification call for unrecognized
  structured medication names would close this gap.
- **The tripwire trades accuracy for safety.** A real deployment would route "unverified"
  issues to a human queue, and the rate is worth monitoring.
- **Timezones.** Structured vitals are UTC; note times are local and unlabeled.
  Comparing them on the same day is approximate (documented tie-break above).
- **Evaluation scale.** 50 templated cases. Before production: a larger labeled set with
  real note variability, and per-rule precision/recall rather than only end-to-end
  match.
- **Harness quirk.** Category match on an empty output scores the 17 READY cases as
  correct (empty list equals empty list), so a run that produces nothing still scores 11%.
