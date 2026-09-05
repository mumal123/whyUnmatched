"""
Run Investigation — WhyUnmatched Pipeline Runner
=================================================
Connects the AI Investigator to real reconciliation exceptions.

Usage:
    python run_investigation.py                    # process all exceptions
    python run_investigation.py --limit 5          # process first 5 only
    python run_investigation.py --limit 1 --verbose  # debug a single one
    python run_investigation.py --dry-run          # show exceptions without calling API
    python run_investigation.py --force            # re-investigate even if already saved
"""

import sys
import io

# Force UTF-8 output on Windows to handle rupee signs and special chars
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import json
import os
import time

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from google import genai

from ai_investigator import (
    DataStore,
    investigate_exception,
    stable_exception_id,
)
from audit_trail import create_audit_entry, save_audit_entry, save_audit_summary, AUDIT_DIR

# Delay between exceptions to stay under free-tier rate limits proactively,
# rather than only reacting to 429s after they happen.
DELAY_BETWEEN_EXCEPTIONS_SECONDS = 4.0


def format_inr(amount: float) -> str:
    """Format amount in Indian Rupee notation."""
    if amount >= 1_00_000:
        return f"Rs.{amount / 1_00_000:.2f}L"
    elif amount >= 1_000:
        return f"Rs.{amount / 1_000:.1f}K"
    else:
        return f"Rs.{amount:.0f}"


def run_dry_run(exceptions_df: pd.DataFrame):
    """Show what would be investigated without calling the API."""
    print(f"\n{'='*60}")
    print(f"DRY RUN -- {len(exceptions_df)} exceptions would be investigated")
    print(f"{'='*60}\n")

    print("Status distribution:")
    print(exceptions_df["status"].value_counts().to_string())

    print(f"\n{'-'*60}")
    for i, (_, row) in enumerate(exceptions_df.iterrows()):
        amounts = []
        if pd.notna(row.get("bank_amount")):
            amounts.append(row["bank_amount"])
        if pd.notna(row.get("settlement_amount")):
            amounts.append(row["settlement_amount"])
        max_amount = max(amounts) if amounts else 0

        exc_id = stable_exception_id(row.to_dict())
        print(
            f"  [{i+1:3d}] {exc_id}  {row['status']:<25s}  "
            f"Amount: {format_inr(max_amount):>12s}"
        )

    print(f"\n{'-'*60}")
    amounts_series = exceptions_df[["bank_amount", "settlement_amount"]].max(axis=1)
    total_exposure = amounts_series.dropna().sum()
    print(f"Total estimated exposure: {format_inr(total_exposure)}")
    print(f"Run without --dry-run to start AI investigation.\n")


def _load_existing_audit_entries(exc_ids: set) -> dict:
    """
    Load audit entries already saved to disk, keyed by exception_id.
    Only returns entries that represent a GENUINE completed investigation —
    an entry saved after a fallback (API error, empty response, invalid JSON)
    always has a reason starting with "Investigation failed:", set by
    _make_fallback_output. Those are excluded here so they get retried on
    the next run instead of being skipped forever.
    """
    existing = {}
    if not os.path.isdir(AUDIT_DIR):
        return existing
    for exc_id in exc_ids:
        path = os.path.join(AUDIT_DIR, f"{exc_id}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    entry = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            reason = entry.get("investigation_result", {}).get("reason", "")
            if reason.startswith("Investigation failed:"):
                continue  # saved failure — retry it, don't skip
            existing[exc_id] = entry
    return existing


def run_investigation(
    data: DataStore,
    exceptions_df: pd.DataFrame,
    api_key: str,
    verbose: bool = False,
    force: bool = False,
):
    """Run AI investigation on all exceptions and produce audit trail."""
    total = len(exceptions_df)
    audit_entries = []
    results = []

    severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    team_counts = {}
    category_counts = {}
    action_counts = {}
    total_money = 0.0
    total_api_time = 0.0

    print(f"\n{'='*60}")
    print(f"INVESTIGATING {total} EXCEPTIONS")
    print(f"{'='*60}\n")

    # Resume support: skip exceptions already saved to disk, unless --force
    exc_id_by_row = {i: stable_exception_id(row.to_dict()) for i, (_, row) in enumerate(exceptions_df.iterrows())}
    existing = {} if force else _load_existing_audit_entries(set(exc_id_by_row.values()))
    if existing and not force:
        print(f"Found {len(existing)} already-investigated exceptions on disk — skipping them.")
        print(f"Use --force to re-investigate everything.\n")

    client = genai.Client(api_key=api_key)

    pipeline_start = time.perf_counter()

    for i, (_, row) in enumerate(exceptions_df.iterrows()):
        exception_dict = row.to_dict()
        status = exception_dict.get("status", "UNKNOWN")
        exc_id = exc_id_by_row[i]

        print(f"[{i+1}/{total}] {exc_id} | {status}...", end=" ", flush=True)

        if not force and exc_id in existing:
            audit = existing[exc_id]
            inv = audit["investigation_result"]
            result = {
                "investigation_result": inv,
                "severity": audit["severity"],
                "assigned_team": audit["assigned_team"],
                "policy_action": audit["policy_action"],
                "tools_called": audit["tools_called"],
                "amount_at_risk": audit["amount_at_risk"],
                "overdue_days": audit["overdue_days"],
                "evidence_warnings": audit["evidence_warnings"],
                "processing_time_seconds": audit["processing_time_seconds"],
            }
            print("already investigated, skipping")
        else:
            try:
                result = investigate_exception(
                    data, exception_dict, client, verbose=verbose
                )
            except Exception as e:
                print(f"ERROR: {str(e)[:60]}")
                result = {
                    "investigation_result": {
                        "exception_type": "UNKNOWN",
                        "risk_factors": [f"Investigation crashed: {str(e)[:100]}"],
                        "evidence_cited": [],
                        "investigation_confidence": 0.0,
                        "suggested_resolution": "MANUAL_REVIEW",
                        "reason": f"Investigation failed with error: {str(e)[:200]}",
                    },
                    "severity": "MEDIUM",
                    "assigned_team": "Finance Review",
                    "policy_action": "MANUAL_REVIEW",
                    "tools_called": [],
                    "amount_at_risk": 0.0,
                    "overdue_days": 0.0,
                    "evidence_warnings": [],
                    "processing_time_seconds": 0.0,
                }

            inv = result["investigation_result"]

            input_records = []
            for col in ("bank_statement_id", "settlement_id", "payment_id"):
                val = exception_dict.get(col)
                if pd.notna(val):
                    input_records.append(str(val))

            audit = create_audit_entry(
                exception_id=exc_id,
                input_records=input_records,
                exception_status=status,
                match_confidence=None,
                tools_called=result["tools_called"],
                investigation_result=inv,
                severity=result["severity"],
                assigned_team=result["assigned_team"],
                policy_action=result["policy_action"],
                amount_at_risk=result["amount_at_risk"],
                overdue_days=result["overdue_days"],
                evidence_warnings=result["evidence_warnings"],
                processing_time_seconds=result["processing_time_seconds"],
            )
            save_audit_entry(audit)

            conf = inv.get("investigation_confidence", 0)
            warn_flag = " [!]" if result["evidence_warnings"] else ""
            print(
                f"severity={result['severity']:<8s} "
                f"conf={conf:.2f}  "
                f"action={result['policy_action']:<15s} "
                f"tools={len(result['tools_called'])}  "
                f"time={result['processing_time_seconds']:.1f}s"
                f"{warn_flag}"
            )

            # Proactive delay to stay under free-tier rate limits
            if i < total - 1:
                time.sleep(DELAY_BETWEEN_EXCEPTIONS_SECONDS)

        inv = result["investigation_result"]
        results.append(result)
        audit_entries.append(audit)

        sev = result["severity"]
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

        team = result["assigned_team"]
        team_counts[team] = team_counts.get(team, 0) + 1

        exc_type = inv.get("exception_type", "UNKNOWN")
        category_counts[exc_type] = category_counts.get(exc_type, 0) + 1

        action = result["policy_action"]
        action_counts[action] = action_counts.get(action, 0) + 1

        total_money += result["amount_at_risk"]
        total_api_time += result["processing_time_seconds"]

    pipeline_elapsed = time.perf_counter() - pipeline_start

    summary_path = save_audit_summary(audit_entries)

    print(f"\n{'='*60}")
    print(f"INVESTIGATION REPORT")
    print(f"{'='*60}")

    print(f"\n[Records Processed]")
    print(f"   Total exceptions:    {total}")
    print(f"   Processing time:     {pipeline_elapsed:.1f}s")
    print(f"   Avg time/exception:  {pipeline_elapsed/total:.1f}s")

    print(f"\n[Financial Impact]")
    print(f"   Total money affected: {format_inr(total_money)}")

    print(f"\n[Severity Breakdown]")
    for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        count = severity_counts.get(sev, 0)
        bar = "#" * count
        print(f"   {sev:<10s} {count:>4d}  {bar}")

    print(f"\n[Exception Categories]")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"   {cat:<25s} {count:>4d}")

    print(f"\n[Team Routing]")
    for team, count in sorted(team_counts.items(), key=lambda x: -x[1]):
        print(f"   {team:<25s} {count:>4d}")

    print(f"\n[Policy Actions]")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"   {action:<25s} {count:>4d}")

    confident_count = sum(
        1 for r in results
        if r["investigation_result"].get("investigation_confidence", 0) >= 0.7
    )
    low_confidence = sum(
        1 for r in results
        if r["investigation_result"].get("investigation_confidence", 0) < 0.5
    )

    print(f"\n[AI Investigation Stats]")
    print(f"   High confidence (>=0.70): {confident_count}/{total}")
    print(f"   Low confidence (<0.50):   {low_confidence}/{total}")
    print(f"   Total API time:           {total_api_time:.1f}s")

    print(f"\n{'-'*60}")
    print(f"UNRESOLVED / LOW CONFIDENCE EXCEPTIONS")
    print(f"{'-'*60}")
    unresolved = [
        (audit_entries[i], results[i])
        for i in range(total)
        if results[i]["investigation_result"].get("investigation_confidence", 0) < 0.5
    ]
    if unresolved:
        for audit, res in unresolved:
            inv = res["investigation_result"]
            print(
                f"   {audit['exception_id']}  "
                f"{audit['exception_status']:<25s}  "
                f"conf={inv.get('investigation_confidence', 0):.2f}  "
                f"action={res['policy_action']:<15s}  "
                f"reason: {inv.get('reason', 'N/A')[:60]}"
            )
    else:
        print("   None -- all exceptions investigated with confidence >= 0.50")

    warned = [
        (audit_entries[i], results[i])
        for i in range(total)
        if results[i]["evidence_warnings"]
    ]
    if warned:
        print(f"\n{'-'*60}")
        print(f"EVIDENCE WARNINGS (cited IDs not found in tool results)")
        print(f"{'-'*60}")
        for audit, res in warned:
            for w in res["evidence_warnings"]:
                print(f"   {audit['exception_id']}: {w}")

    print(f"\n[Audit Trail] {summary_path}")
    print(f"{'='*60}\n")

    return results, audit_entries


def main():
    parser = argparse.ArgumentParser(
        description="WhyUnmatched -- AI Exception Investigation Pipeline"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of exceptions to investigate (default: all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print tool calls and results for debugging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be investigated without calling API",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-investigate exceptions even if already saved to the audit trail",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key and not args.dry_run:
        print("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        print("Get a free key at: https://aistudio.google.com/apikey")
        print("Then: set GEMINI_API_KEY=your_key_here")
        print("Or use --dry-run to preview without API calls.")
        exit(1)

    print("Loading data...")
    data = DataStore()

    all_exceptions = data.final_recon[data.final_recon["status"] != "MATCH"].copy()
    print(f"Found {len(all_exceptions)} exceptions in final_reconciliation.csv")

    if args.limit:
        all_exceptions = all_exceptions.head(args.limit)
        print(f"Limited to first {args.limit}")

    if args.dry_run:
        run_dry_run(all_exceptions)
        return

    run_investigation(data, all_exceptions, api_key, verbose=args.verbose, force=args.force)


if __name__ == "__main__":
    main()