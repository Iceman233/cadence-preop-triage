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
8. **Document identity is deterministic; document content is extracted.** Whether a
   document *is* an H&P or a consent comes from its title or opening heading
   (typo-tolerant: "History and Pyhsical"). In live runs the model labeled pre-admission
   and anesthesia notes as H&Ps, which would falsely satisfy Rule 1, the dangerous
   direction. A real H&P filed under an unrelated title would now be missed, which
   fails safe (follow-up).

## Results

Sample set (50 cases), `make evals-local` and `make robustness`, model `gpt-4.1-mini`:

| Set | Heuristic score | Heuristic decision | LLM score | LLM decision | False READY (both) |
|---|---|---|---|---|---|
| Original | 100.0 | 100% | 100.0 | 100% | 0 |
| Paraphrased vitals | 57.3 | 54% | 100.0 | 100% | 0 |
| Paraphrased consent | 63.3 | 66% | 100.0 | 100% | 0 |
| Paraphrased plans | 93.3 | 92% | 100.0 | 100% | 0 |
| Paraphrased medications | 99.3 | 100% | 100.0 | 100% | 0 |
| All paraphrases | 53.3 | 54% | 100.0 | 100% | 0 |

The starter's hosted OpenAI Evals run agrees on the original set (50/50 passed).
Determinism (`make determinism`, case 0, 10 runs): 100% exact-output match for the
heuristic, and for the LLM both with the cache and with it disabled
(`TRIAGE_LLM_CACHE=0`), because extraction variation does not reach deterministic
rendering. That is one case; the per-document cache is what guarantees repeatability.
LLM cost: one call per document (about 4 s each, run 4 at a time), about 3.5 minutes for
the 50 cases uncached.

**How to read this.** The heuristic's 100% is in-sample: the notes are templated and I
calibrated it on these 50 cases. `scripts/perturb.py` rewrites the decision-relevant
phrases with the same meaning (labels still apply), and the heuristic drops to 53%.
That gap is what the LLM extractor is for, and it closes it. The LLM prompt was tuned
only on failures on the original set; the paraphrase sets were never used for tuning,
so they are a held-out check (though still synthetic, and written by me).

**What the evaluations caught** (each fixed and covered by a test):

- *Tripwire with the extractor's blind spots.* The first tripwire reused the
  extractor's regex, so "blood pressure measured at 157 over 117" slipped past both:
  **4 false READYs** on paraphrased vitals. The tripwire is now independently broader.
- *Right decision, wrong evidence.* Case 32 was NOT_CLEARED but cited the arrival BP
  instead of the later recheck, because "at 09:38:" was not parsed as a time.
- *Live LLM failure modes*, found by verification and tripwires rather than by luck.
  The first LLM run scored 67% with 0 false READYs, because every model mistake was
  flagged instead of trusted:
  - the model mangled "°C" when copying quotes ("\x00b0C", "\x176"), so verbatim quote
    checks failed. The model now receives ASCII text, and quotes are matched on letters
    and digits, then mapped back to the document's exact wording for evidence;
  - one entry carried both a BP and a temperature (now split in code);
  - quotes stitched together from separate fragments of a line (prompt: one
    contiguous span; time goes in `time_of_day`);
  - pre-admission and anesthesia notes classified as H&Ps (decision 8).

## Assumptions

| Area | Assumption |
|---|---|
| Review time | `metadata.submission_received_at`; readings dated after it are ignored. |
| Windows | Calendar days, inclusive (exactly 30 passes). Labs/H&Ps dated after the procedure do not count. |
| H&P | Any H&P within 30 days counts (the policy does not require a "pre-op" H&P; no non-pre-op H&P in the data falls in the window). Identified by title or opening heading (typo-tolerant), not by model judgment (decision 8). If the in-text date of service is older than the document date, the older date is used. |
| Consent | Only the **patient's** signature counts, never the physician attestation. A newer consent for the same procedure supersedes an older one. Only an actual procedure mismatch fails; a consent naming no procedure is accepted (as in the prompt's example). |
| Labs | Usable statuses: final, amended, corrected, or missing. Matched by code or display (CBC/LAB-CBC/Hemogram). A BMP is not a CMP. |
| Anticoagulants | Active if listed `active: true` or a note says the patient takes it, unless a later fact says stopped. `active: null` or "unable to confirm" → `MISSING_REQUIRED_DATA`. A plan must name the same drug (brand → generic) and give a concrete pre-op **and** post-op step. Bridging agents (enoxaparin) need no plan of their own. Antiplatelets never need one. |
| Vitals | BP and temperature evaluated independently. Order: date, then time of day; a same-day note reading without a time sorts after timed readings; on an exact tie the more severe reading wins. Celsius converted; a unitless value under 50 is read as Celsius. No BP or no temperature → `MISSING_REQUIRED_DATA`. |
| Precedence | NOT_CLEARED > NEEDS_FOLLOW_UP > READY; all issues are listed regardless. |

## Limitations and next steps

- **Synthetic evaluation.** Both the sample and the paraphrases are templated text. The
  LLM's 100% shows the pipeline works end to end, not that it is production-accurate.
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
