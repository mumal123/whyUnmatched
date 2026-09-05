
"""
Generate Priority Queue — WhyUnmatched
=======================================
Reads the audit trail produced by run_investigation.py and ranks every
exception by cashflow impact, so a reviewer sees what needs attention first
instead of N equally-red rows.

Usage:
    python generate_priority_queue.py
    python generate_priority_queue.py --top 10
"""

import argparse
import csv
import json
import os

from policy_engine import rank_exceptions, POLICY_VERSION

AUDIT_DIR = os.path.join("data", "audit_trail")
AUDIT_SUMMARY_PATH = os.path.join(AUDIT_DIR, "summary.json")
OUTPUT_CSV_PATH = os.path.join("data", "priority_queue.csv")


def format_inr(amount: float) -> str:
    if amount >= 1_00_000:
        return f"Rs.{amount / 1_00_000:.2f}L"
    elif amount >= 1_000:
        return f"Rs.{amount / 1_000:.1f}K"
    return f"Rs.{amount:.0f}"


def load_audit_entries() -> list[dict]:
    """
    Load every individual exception JSON file from data/audit_trail/, rather
    than relying solely on summary.json — summary.json is only written after
    a run completes in full, so an interrupted run (rate limit, Ctrl+C, etc.)
    would otherwise leave nothing to rank even though real results exist on
    disk. This picks up whatever has actually been investigated so far.
    """
    if not os.path.isdir(AUDIT_DIR):
        raise FileNotFoundError(
            f"{AUDIT_DIR} not found — run run_investigation.py first."
        )

    entries = []
    for filename in os.listdir(AUDIT_DIR):
        if filename == "summary.json" or not filename.endswith(".json"):
            continue
        path = os.path.join(AUDIT_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                entries.append(json.load(fh))
        except (json.JSONDecodeError, OSError):
            continue

    if not entries:
        raise FileNotFoundError(
            f"No exception files found in {AUDIT_DIR} — run run_investigation.py first."
        )

    return entries


def save_priority_queue_csv(ranked: list[dict]) -> str:
    fieldnames = [
        "priority_rank",
        "priority_score",
        "exception_id",
        "exception_status",
        "severity",
        "assigned_team",
        "policy_action",
        "amount_at_risk",
        "overdue_days",
        "investigation_confidence",
    ]
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for entry in ranked:
            writer.writerow({k: entry.get(k, "") for k in fieldnames})
    return OUTPUT_CSV_PATH


def print_top(ranked: list[dict], top_n: int):
    print(f"\n{'='*70}")
    print(f"PRIORITY QUEUE — top {min(top_n, len(ranked))} of {len(ranked)} exceptions")
    print(f"(policy_version {POLICY_VERSION})")
    print(f"{'='*70}\n")

    skipped = 0
    for entry in ranked[:top_n]:
        required = ("policy_action", "severity", "assigned_team", "exception_id", "exception_status")
        missing = [k for k in required if k not in entry]
        if missing:
            skipped += 1
            print(
                f"  [SKIPPED] {entry.get('exception_id', '?')} — malformed audit entry, "
                f"missing fields: {missing}. This file may predate a schema fix; "
                f"consider deleting and re-investigating with --force.\n"
            )
            continue

        amt = entry.get("amount_at_risk", 0) or 0
        overdue = entry.get("overdue_days", 0) or 0
        conf = entry.get("investigation_confidence") or 0.0
        print(
            f"Priority {entry['priority_rank']:>3d}  "
            f"[{entry['severity']:<8s}]  "
            f"{format_inr(amt):>10s} at risk"
            + (f", {overdue:.0f}d overdue" if overdue > 0 else "")
            + f"  |  conf={conf:.2f}  ->  {entry['policy_action']:<15s}  "
            f"({entry['assigned_team']})"
        )
        print(f"           {entry['exception_id']}  [{entry['exception_status']}]")

    print(f"\n{'-'*70}")
    total_at_top = sum((e.get("amount_at_risk") or 0) for e in ranked[:top_n])
    print(f"Total exposure in top {min(top_n, len(ranked))}: {format_inr(total_at_top)}")
    if skipped:
        print(f"({skipped} malformed entr{'y' if skipped == 1 else 'ies'} skipped — see above)")
    print(f"Full ranked list written to {OUTPUT_CSV_PATH}\n")


def main():
    parser = argparse.ArgumentParser(description="Rank reconciliation exceptions by cashflow impact")
    parser.add_argument("--top", type=int, default=15, help="How many top-priority exceptions to print (default: 15)")
    args = parser.parse_args()

    entries = load_audit_entries()
    if not entries:
        print("No audit entries found — nothing to rank.")
        return

    total_exceptions_path = os.path.join("data", "final_reconciliation.csv")
    if os.path.exists(total_exceptions_path):
        import csv as _csv
        with open(total_exceptions_path, newline="", encoding="utf-8") as fh:
            total_in_dataset = sum(1 for row in _csv.DictReader(fh) if row.get("status") != "MATCH")
        if len(entries) < total_in_dataset:
            print(f"\nNote: {len(entries)} of {total_in_dataset} total exceptions investigated so far "
                  f"(run was interrupted or partial — resume with run_investigation.py to continue).")

    ranked = rank_exceptions(entries)
    save_priority_queue_csv(ranked)
    print_top(ranked, args.top)


if __name__ == "__main__":
    main()