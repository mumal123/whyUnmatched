# WhyUnmatched

WhyUnmatched is a reconciliation-exception demo for Razorpay Buildathon. It matches bank credits to settlements, classifies the leftovers, prioritises risk, and records a reviewable audit trail. The included data is synthetic; live Test Mode ingestion is optional.

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

## Optional AI investigation

Copy the template, then add your own key locally:

```powershell
Copy-Item .env.example .env
# Edit .env locally and set GEMINI_API_KEY
python run_investigation.py --limit 5
python generate_priority_queue.py
```

`run_investigation.py` is deliberately not part of the default demo: it calls an external model and may incur rate limits or cost. The policy engine, not the model, chooses the routing/action.

## Optional Razorpay Test Mode ingestion

Set `RAZORPAY_KEY_ID` and `RAZORPAY_KEY_SECRET` in the local `.env`, then run:

```powershell
python fetch_razorpay_data.py --count 50
```

This writes to `data_razorpay_live/`, which is ignored by Git. The basic `/settlements` endpoint returns settlement batches rather than an individual payment mapping, so these files demonstrate ingestion and schema normalization—not a production-ready reconciliation mapping. A production implementation should ingest Razorpay’s settlement/reconciliation report or Transactions API data.

## Security and data handling

- Do not commit `.env`, API keys, or merchant-specific exports. `.gitignore` excludes them and `.env.example` contains only blank placeholders.
- If a real credential was ever committed or pasted into a shared channel, rotate it in the provider dashboard immediately.
- The audit trail may contain financial identifiers. Treat `data/audit_trail/` as demo data only; use controlled storage, access controls, and retention policies in production.

## Verification

Run these after installation:

```powershell
python -m compileall -q .
python run_pipeline.py
python evaluate.py
python dashboard.py
```

`evaluate.py` reports precision and recall against the generated ground truth. The final command should print a localhost URL; load it in a browser to validate the dashboard.
