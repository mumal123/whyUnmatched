"""
Policy Engine — WhyUnmatched
============================
Deterministic rules that decide severity, team routing, and the final action
for a reconciliation exception. This module is intentionally NOT the LLM —
everything here is a plain, versioned, inspectable rule so it can be shown,
demoed, and audited on its own, independent of any AI investigation.

ai_investigator.py should import compute_severity, route_to_team, and
determine_policy_action from HERE rather than defining them inline, so there
is exactly one place these rules live.
"""

POLICY_VERSION = "v1.0"


# ---------------------------------------------------------------------------
# Severity rules — explicit, ordered, checked top to bottom
# ---------------------------------------------------------------------------

SEVERITY_RULES = [
    {"level": "CRITICAL", "condition": "amount_at_risk > 500000 or overdue_days > 5"},
    {"level": "HIGH",     "condition": "amount_at_risk > 100000 or overdue_days > 3"},
    {"level": "MEDIUM",   "condition": "amount_at_risk > 10000"},
    {"level": "LOW",      "condition": "everything else"},
]


def compute_severity(amount_at_risk: float, overdue_days: float, exception_type: str) -> str:
    """
    CRITICAL: amount > 5L or overdue > 5 days
    HIGH:     amount > 1L or overdue > 3 days
    MEDIUM:   amount > 10K
    LOW:      everything else
    """
    if amount_at_risk > 500_000 or overdue_days > 5:
        return "CRITICAL"
    if amount_at_risk > 100_000 or overdue_days > 3:
        return "HIGH"
    if amount_at_risk > 10_000:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Team routing — explicit mapping, no inference
# ---------------------------------------------------------------------------

TEAM_ROUTING = {
    "MISSING_SETTLEMENT": "Gateway Support",
    "MISSING_BANK_CREDIT": "Gateway Support",
    "AMOUNT_MISMATCH": "Finance Review",
    "UNMATCHED_CREDIT": "Operations",
    "DUPLICATE_PAYMENT": "Finance Review",
    "DUPLICATE": "Finance Review",
    "PARTIAL_PAYMENT": "Customer Support",
}


def route_to_team(exception_type: str) -> str:
    """Deterministic mapping of exception type -> responsible team."""
    return TEAM_ROUTING.get(exception_type, "Finance Review")


# ---------------------------------------------------------------------------
# Policy action — the actual decision, gated on confidence + exposure
# ---------------------------------------------------------------------------

POLICY_ACTION_RULES = [
    {"action": "FINANCE_REVIEW", "condition": "confidence >= 0.85 and exposure < threshold"},
    {"action": "SENIOR_REVIEW",  "condition": "confidence >= 0.85 and exposure >= threshold"},
    {"action": "HOLD_FOR_REVIEW", "condition": "confidence >= 0.50"},
    {"action": "MANUAL_REVIEW",  "condition": "confidence < 0.50"},
]

DEFAULT_EXPOSURE_THRESHOLD = 100_000


def determine_policy_action(
    investigation_confidence: float,
    amount_at_risk: float,
    exposure_threshold: float = DEFAULT_EXPOSURE_THRESHOLD,
) -> str:
    """
    Policy rules (v1.0):
        confidence >= 0.85 AND exposure < threshold   -> FINANCE_REVIEW
        confidence >= 0.85 AND exposure >= threshold  -> SENIOR_REVIEW
        confidence >= 0.50                            -> HOLD_FOR_REVIEW
        confidence < 0.50                             -> MANUAL_REVIEW

    The LLM never picks this — it only supplies investigation_confidence and
    the amount_at_risk is computed deterministically in ai_investigator.py.
    """
    if investigation_confidence >= 0.85:
        if amount_at_risk < exposure_threshold:
            return "FINANCE_REVIEW"
        return "SENIOR_REVIEW"
    if investigation_confidence >= 0.50:
        return "HOLD_FOR_REVIEW"
    return "MANUAL_REVIEW"


# ---------------------------------------------------------------------------
# Priority ranking — cashflow-impact prioritisation
#
# Not every exception deserves equal attention. A ₹15 discrepancy can wait;
# a ₹5 lakh missing settlement cannot. This ranks the full exception set so
# a reviewer sees "what needs attention first", not 200 equally-red rows.
# ---------------------------------------------------------------------------

SEVERITY_WEIGHT = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}


def compute_priority_score(
    amount_at_risk: float,
    overdue_days: float,
    severity: str,
    investigation_confidence: float,
) -> float:
    """
    Higher score = investigate first.

    Weighting rationale (documented, not hidden):
      - amount_at_risk dominates (money at stake matters most; log-scaled
        so a ₹50L exception doesn't make every smaller one invisible)
      - overdue_days adds urgency on top of amount
      - severity is a coarse multiplier reflecting the policy tier
      - LOW confidence investigations get a priority bump — an exception
        the AI wasn't sure about needs human eyes sooner, not later,
        since an unconfident diagnosis is itself a risk signal
    """
    import math

    amount_component = math.log10(max(amount_at_risk, 1) + 1) * 10
    overdue_component = min(overdue_days, 30) * 2
    severity_multiplier = SEVERITY_WEIGHT.get(severity, 1)
    confidence_penalty = (1.0 - investigation_confidence) * 15

    return round(
        (amount_component + overdue_component) * severity_multiplier + confidence_penalty,
        2,
    )


def rank_exceptions(audit_entries: list[dict]) -> list[dict]:
    """
    Takes a list of audit entries (as produced by audit_trail.create_audit_entry)
    and returns them sorted by priority_score, descending, with the score and
    its rank attached. Does not mutate the input entries.
    """
    ranked = []
    for entry in audit_entries:
        score = compute_priority_score(
            amount_at_risk=entry.get("amount_at_risk", 0) or 0,
            overdue_days=entry.get("overdue_days", 0) or 0,
            severity=entry.get("severity", "LOW"),
            investigation_confidence=entry.get("investigation_confidence") or 0.0,
        )
        ranked.append({**entry, "priority_score": score})

    ranked.sort(key=lambda e: e["priority_score"], reverse=True)
    for i, entry in enumerate(ranked):
        entry["priority_rank"] = i + 1

    return ranked