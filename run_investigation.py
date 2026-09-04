"""
Run Investigation — WhyUnmatched Pipeline Runner
=================================================
Connects the AI Investigator to real reconciliation exceptions.

Usage:
    python run_investigation.py                    # process all exceptions
    python run_investigation.py --limit 5          # process first 5 only
    python run_investigation.py --limit 1 --verbose  # debug a single one
    python run_investigation.py --dry-run          # show exceptions without calling API
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
from audit_trail import create_audit_entry, save_audit_entry, save_audit_summary


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


def run_investigation(
    data: DataStore,
    exceptions_df: pd.DataFrame,
    api_key: str,
    verbose: bool = False,
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

    # Priority 4 — Create Gemini client ONCE, reuse for all exceptions
    client = genai.Client(api_key=api_key)

    pipeline_start = time.perf_counter()

    for i, (_, row) in enumerate(exceptions_df.iterrows()):
        exception_dict = row.to_dict()
        status = exception_dict.get("status", "UNKNOWN")

        # Priority 9 — Stable, reproducible exception ID
        exc_id = stable_exception_id(exception_dict)

        # Priority 17 — Clear per-investigation logging
        print(f"[{i+1}/{total}] {exc_id} | {status}...", end=" ", flush=True)

        # Priority 5/12 — Per-exception error handling (never crash entire run)
        try:
            result = investigate_exception(
                data, exception_dict, client, verbose=verbose
            )
        except Exception as e:
            # Safe fallback for unexpected errors
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
        results.append(result)

        # Track metrics
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

        # Collect input record IDs for audit
        input_records = []
        for col in ("bank_statement_id", "settlement_id", "payment_id"):
            val = exception_dict.get(col)
            if pd.notna(val):
                input_records.append(str(val))

        # Create audit entry
        audit = create_audit_entry(
            exception_id=exc_id,
            input_records=input_records,
            exception_status=status,
            match_confidence=None,  # fuzzy matcher doesn't export this yet
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
        audit_entries.append(audit)
        save_audit_entry(audit)

        # Priority 17 — Per-investigation summary line
        conf = inv.get("investigation_confidence", 0)
        warn_flag = " [!]" if result["evidence_warnings"] else ""
        print(
            f"severity={sev:<8s} "
            f"conf={conf:.2f}  "
            f"action={action:<15s} "
            f"tools={len(result['tools_called'])}  "
            f"time={result['processing_time_seconds']:.1f}s"
            f"{warn_flag}"
        )

    pipeline_elapsed = time.perf_counter() - pipeline_start

    # Save consolidated summary
    summary_path = save_audit_summary(audit_entries)

    # ---------------------------------------------------------------
    # Print report
    # ---------------------------------------------------------------
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

    # AI resolution stats
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

    # Show unresolved (low confidence) — NEVER hide these (Priority 15/18)
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

    # Show evidence warnings
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
    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key and not args.dry_run:
        print("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        print("Get a free key at: https://aistudio.google.com/apikey")
        print("Then: set GEMINI_API_KEY=your_key_here")
        print("Or use --dry-run to preview without API calls.")
        exit(1)

    # Load data
    print("Loading data...")
    data = DataStore()

    # Get exceptions — Priority 6/16: ONLY non-MATCH records
    all_exceptions = data.final_recon[data.final_recon["status"] != "MATCH"].copy()
    print(f"Found {len(all_exceptions)} exceptions in final_reconciliation.csv")

    if args.limit:
        all_exceptions = all_exceptions.head(args.limit)
        print(f"Limited to first {args.limit}")

    if args.dry_run:
        run_dry_run(all_exceptions)
        return

    run_investigation(data, all_exceptions, api_key, verbose=args.verbose)


if __name__ == "__main__":
    main()
