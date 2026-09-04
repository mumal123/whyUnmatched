"""
Audit Trail — WhyUnmatched
==========================
Creates replayable, timestamped records for every exception investigation.
Each exception gets an individual JSON file. A consolidated summary is also written.

The AI Investigator never writes here directly — only the pipeline runner does.
"""

import json
import os
from datetime import datetime, timezone


AUDIT_DIR = os.path.join("data", "audit_trail")


def _ensure_dir():
    os.makedirs(AUDIT_DIR, exist_ok=True)


def create_audit_entry(
    exception_id: str,
    input_records: list[str],
    exception_status: str,
    match_confidence: float | None,
    tools_called: list[str],
    investigation_result: dict,
    severity: str,
    assigned_team: str,
    policy_action: str,
    amount_at_risk: float,
    overdue_days: float,
    evidence_warnings: list[str],
    processing_time_seconds: float,
) -> dict:
    """Build a complete audit entry dict (does NOT write to disk)."""
    return {
        "exception_id": exception_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_records": input_records,
        "exception_status": exception_status,
        "match_confidence": match_confidence,
        "tools_called": tools_called,
        "investigation_result": investigation_result,
        "investigation_confidence": investigation_result.get(
            "investigation_confidence"
        ),
        "severity": severity,
        "assigned_team": assigned_team,
        "policy_action": policy_action,
        "amount_at_risk": round(amount_at_risk, 2),
        "overdue_days": round(overdue_days, 1),
        "evidence_warnings": evidence_warnings,
        "processing_time_seconds": round(processing_time_seconds, 3),
        "policy_version": "v1.0",
    }


def save_audit_entry(entry: dict) -> str:
    """Write a single audit entry to its own JSON file. Returns the filepath."""
    _ensure_dir()
    filepath = os.path.join(AUDIT_DIR, f"{entry['exception_id']}.json")
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(entry, fh, indent=2, ensure_ascii=False)
    return filepath


def save_audit_summary(entries: list[dict]) -> str:
    """Write the consolidated summary of all audit entries."""
    _ensure_dir()

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy_version": "v1.0",
        "total_exceptions": len(entries),
        "by_severity": {},
        "by_team": {},
        "by_exception_type": {},
        "by_policy_action": {},
        # Priority 8 — aggregate actual exposure from entries
        "total_money_affected": 0.0,
        "entries": entries,
    }

    for e in entries:
        sev = e.get("severity", "UNKNOWN")
        summary["by_severity"][sev] = summary["by_severity"].get(sev, 0) + 1

        team = e.get("assigned_team", "UNKNOWN")
        summary["by_team"][team] = summary["by_team"].get(team, 0) + 1

        etype = e.get("exception_status", "UNKNOWN")
        summary["by_exception_type"][etype] = (
            summary["by_exception_type"].get(etype, 0) + 1
        )

        action = e.get("policy_action", "UNKNOWN")
        summary["by_policy_action"][action] = (
            summary["by_policy_action"].get(action, 0) + 1
        )

        # Priority 8 — use the actual amount_at_risk from each entry
        summary["total_money_affected"] += e.get("amount_at_risk", 0)

    summary["total_money_affected"] = round(summary["total_money_affected"], 2)

    filepath = os.path.join(AUDIT_DIR, "summary.json")
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    return filepath
