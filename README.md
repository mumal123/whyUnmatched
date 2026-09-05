# WhyUnmatched

WhyUnmatched is a reconciliation-exception demo for the Razorpay Buildathon. It matches bank credits to settlements, classifies the leftovers, prioritises risk, and records a reviewable audit trail. The included data is synthetic; live Razorpay Test Mode ingestion is optional.

**The core idea:** existing reconciliation tools tell you what didn't match. This tells you *why* it didn't match, how much money is at risk, who should act, and what the safest next step is — while keeping every dollar-moving decision in deterministic code, never in the model.

## Architecture

```
                Bank Statements (no payment_id)          Settlements
                          │                                    │
                          └───────────────┬────────────────────┘
                                          ▼
                              EXACT MATCH (amount + time window)
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                    ▼
                     MATCHED                            LEFTOVER
                                                             │
                                                             ▼
                                          FUZZY MATCH (tolerance + fraction detection)
                                                             │
                        ┌───────────┬───────────┬───────────┼────────────┐
                        ▼           ▼           ▼           ▼            ▼
                PARTIAL_PAYMENT  AMOUNT_    DUPLICATE  UNMATCHED_   MISSING_
                                 MISMATCH              CREDIT      SETTLEMENT
                        │           │           │           │            │
                        └───────────┴───────────┴───────────┴────────────┘
                                                 │
                                                 ▼
                                   EVALUATE (precision / recall / exception list
                                              against synthetic ground truth)
                                                 │
                                                 ▼
                              AI INVESTIGATOR  (optional, Gemini + read-only tools:
                                lookup_records, calculate_exposure, check_timeline)
                                    — diagnoses WHY, never decides WHAT —
                                                 │
                                                 ▼
                                POLICY ENGINE  (deterministic — NOT the LLM)
                                 severity → assigned_team → policy_action
                                                 │
                                                 ▼
                              PRIORITY QUEUE  (ranked by cashflow impact,
                                    not "200 equally-red rows")
                                                 │
                                                 ▼
                                  AUDIT TRAIL (one JSON per exception)
                                                 │
                                                 ▼
                                      DASHBOARD (read-only, local)
```

**The one architectural decision that matters most:** the AI Investigator gathers evidence and produces a diagnosis (`exception_type`, `risk_factors`, `evidence_cited`, `reason`) — it never picks the final action. Severity, team routing, and the policy action (`FINANCE_REVIEW` / `SENIOR_REVIEW` / `HOLD_FOR_REVIEW` / `MANUAL_REVIEW`) are computed by plain, versioned rules in `policy_engine.py`, using only the confidence score and financial exposure the LLM supplied — not its opinion on what should happen. This is what makes the system auditable even when the model is wrong: a bad diagnosis can escalate too cautiously, but it can never take an action the deterministic layer wouldn't otherwise allow.

`nl_query.py` reuses the exact same three tools conversationally — ask "why is settlement X overdue?" and it answers from real tool calls, never invented data.

## Project structure

| File | Role |
|---|---|
| `generate_data.py` | Builds synthetic payments/settlements/bank statements **with a known ground-truth answer key**, then injects realistic exceptions (missing settlements, partial payments, amount mismatches, duplicates, amount collisions) |
| `exact_match.py` | Deterministic exact matching — amount + settlement-delay window, vectorized (no row loops) |
| `fuzzy_match.py` | Tolerance-based matching for what's left — partial payments, small drift, duplicate detection against everything already accounted for |
| `evaluate.py` | Pairwise precision/recall against ground truth + a full expected-vs-predicted exception list — never scores off a single cherry-picked match |
| `policy_engine.py` | The deterministic rules — severity, team routing, policy action, and cashflow-impact priority ranking. Single source of truth; nothing here is the LLM |
| `ai_investigator.py` | Gemini-based investigator with three read-only tools (`lookup_records`, `calculate_exposure`, `check_timeline`); imports its policy decisions from `policy_engine.py` |
| `run_investigation.py` | Batch runner over all exceptions — rate-limit retry/backoff, resume-on-failure, per-exception audit logging |
| `generate_priority_queue.py` | Ranks whatever's been investigated (even a partial run) by cashflow impact into `data/priority_queue.csv` |
| `nl_query.py` | Conversational Q&A over the same tool layer — evidence-grounded, multi-turn |
| `audit_trail.py` | One JSON record per exception investigated, plus a consolidated summary |
| `fetch_razorpay_data.py` | Optional real Razorpay Test Mode ingestion (Payments + Settlements APIs), normalized into the same schema the matcher expects |
| `dashboard.py` | Local, read-only view of exception counts, exposure, status mix, and the priority queue |
| `run_pipeline.py` | Runs the deterministic stages (generate → exact → fuzzy → evaluate) in one command |

## What the demo shows

- Deterministic exact matching by amount and settlement window.
- Fuzzy classification of partial payments, small amount mismatches, duplicate credits, missing credits, and missing settlements.
- A separate, versioned policy engine for severity, routing, and suggested review action.
- Optional Gemini investigations with read-only evidence tools and JSON audit records.
- A local read-only dashboard for presenting exception count, exposure, status mix, and the priority queue.

## Quick start

Use Python 3.10+ in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_pipeline.py
python dashboard.py
```

Open `http://127.0.0.1:8000`. The dashboard reads `data/final_reconciliation.csv` and, when present, `data/priority_queue.csv`; it never reads or returns `.env` values.

The deterministic pipeline requires no API keys. To create fresh synthetic demo data, run `python generate_data.py` before `python run_pipeline.py`.

## Evaluation

`evaluate.py` measures the reconciliation engine pairwise against the synthetic ground truth — throughput, precision, recall, and a full expected-vs-predicted exception list, not a single cherry-picked match. On the reference dataset (1,000 transactions, including 10 deliberately injected amount-collision pairs designed to be genuinely ambiguous):

- **Precision ≈ 99.2%, Recall = 100%** — every real exception is found; the small number of false positives are exclusively same-amount, same-window collisions, which is the one case amount+time matching alone cannot fully resolve without an additional signal.
- Throughput comfortably exceeds 100,000 records/sec for exact matching (vectorized, not row-by-row).
- Every miss is logged with its expected vs. predicted pairing and a reason — see `data/exception_list.csv`.

## Optional AI investigation

Copy the template, then add your own key locally:

```powershell
Copy-Item .env.example .env
# Edit .env locally and set GEMINI_API_KEY
python run_investigation.py --limit 5
python generate_priority_queue.py
```

`run_investigation.py` is deliberately not part of the default demo: it calls an external model and may incur rate limits or cost. The policy engine, not the model, chooses the routing/action. It also resumes safely — a saved failure (e.g. a rate-limited call) is retried on the next run rather than skipped forever, and `generate_priority_queue.py` ranks whatever has been investigated so far, even a partial run.

## Optional Razorpay Test Mode ingestion

Set `RAZORPAY_KEY_ID` and `RAZORPAY_KEY_SECRET` in the local `.env`, then run:

```powershell
python fetch_razorpay_data.py --count 50
```

This writes to `data_razorpay_live/`, which is ignored by Git. The basic `/settlements` endpoint returns settlement batches rather than an individual payment mapping, so these files demonstrate ingestion and schema normalization — not a production-ready reconciliation mapping. A production implementation should ingest Razorpay's settlement/reconciliation report or Transactions API data. Precision/recall evaluation still runs against the synthetic ground truth, since measuring accuracy requires a known-correct answer that real transaction data doesn't carry on its own.

## Security and data handling

- Do not commit `.env`, API keys, or merchant-specific exports. `.gitignore` excludes them and `.env.example` contains only blank placeholders.
- If a real credential was ever committed or pasted into a shared channel, rotate it in the provider dashboard immediately.
- The audit trail may contain financial identifiers. Treat `data/audit_trail/` as demo data only; use controlled storage, access controls, and retention policies in production.

## Known limitations

- Settlement matching assumes one settlement per payment. Real Razorpay settlements can batch multiple payments into a single bank transfer — a documented simplification, not an oversight (see `fetch_razorpay_data.py` docstring).
- The AI Investigator's free-tier daily quota (20 requests/day on `gemini-2.5-flash`) limits how many exceptions can be investigated per day without enabling billing. The deterministic pipeline (matching, evaluation, priority ranking) has no such limit.

## Verification

Run these after installation:

```powershell
python -m compileall -q .
python run_pipeline.py
python evaluate.py
python dashboard.py
```

`evaluate.py` reports precision and recall against the generated ground truth. The final command should print a localhost URL; load it in a browser to validate the dashboard.