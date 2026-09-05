"""Run deterministic reconciliation stages and write demo-ready CSV outputs.

This command never calls an LLM or external API. It is therefore safe to run
with the included synthetic data and needs no credentials.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fuzzy_match import fuzzy_match
from reconciliation.excat_matcher import exact_match


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
ALLOWED_STATUSES = {
    "MATCH",
    "MISSING_SETTLEMENT",
    "AMOUNT_MISMATCH",
    "DUPLICATE",
    "UNMATCHED_CREDIT",
    "PARTIAL_PAYMENT",
    "MISSING_BANK_CREDIT",
}


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
    return frame


def run(data_dir: Path) -> pd.DataFrame:
    """Execute exact + fuzzy matching and return the combined result."""
    bank = _read_csv(data_dir / "bank_statements_engine_input.csv", {"bank_statement_id", "amount", "time"})
    settlements = _read_csv(data_dir / "settlements.csv", {"settlement_id", "payment_id", "settlement_amount", "time"})
    payments = _read_csv(data_dir / "payments.csv", {"payment_id"})

    exact_result, exact_elapsed = exact_match(bank, settlements)
    exact_result.to_csv(data_dir / "exact_reconciliation.csv", index=False)

    bank_leftover = bank[bank["bank_statement_id"].isin(
        exact_result.loc[exact_result["status"] == "UNMATCHED_BANK_CREDIT", "bank_statement_id"].dropna()
    )]
    settlement_leftover = settlements[settlements["settlement_id"].isin(
        exact_result.loc[exact_result["status"] == "MISSING_SETTLEMENT", "settlement_id"].dropna()
    )]
    fuzzy_result, fuzzy_elapsed = fuzzy_match(bank_leftover, settlement_leftover)

    # A previously accounted-for amount reveals duplicate credits among fuzzy leftovers.
    accounted = pd.concat([
        exact_result.loc[exact_result["status"] == "MATCH", "settlement_amount"],
        fuzzy_result.loc[fuzzy_result["status"].isin({"MATCH", "AMOUNT_MISMATCH", "PARTIAL_PAYMENT"}), "settlement_amount"],
    ]).dropna()
    unmatched = fuzzy_result["status"] == "UNMATCHED_CREDIT"
    fuzzy_result.loc[unmatched & fuzzy_result["bank_amount"].apply(
        lambda amount: ((accounted - amount).abs() <= 2.0).any()
    ), "status"] = "DUPLICATE"

    settled_payment_ids = set(settlements["payment_id"].dropna())
    missing_payments = payments[~payments["payment_id"].isin(settled_payment_ids)]
    missing_result = pd.DataFrame({
        "bank_statement_id": [None] * len(missing_payments),
        "settlement_id": [None] * len(missing_payments),
        "bank_amount": [None] * len(missing_payments),
        "settlement_amount": [None] * len(missing_payments),
        "status": "MISSING_SETTLEMENT",
        "payment_id": missing_payments["payment_id"].values,
    })

    final = pd.concat([
        exact_result.loc[exact_result["status"] == "MATCH"],
        fuzzy_result,
        missing_result,
    ], ignore_index=True)
    final.to_csv(data_dir / "final_reconciliation.csv", index=False)

    unexpected = set(final["status"].dropna()) - ALLOWED_STATUSES
    if unexpected or final["status"].isna().any():
        raise AssertionError(f"Invalid reconciliation statuses: {sorted(unexpected)}")

    print(f"Exact matching: {len(exact_result)} records in {exact_elapsed:.3f}s")
    print(f"Fuzzy matching: {len(fuzzy_result)} records in {fuzzy_elapsed:.3f}s")
    print(f"Wrote {len(final)} rows to {data_dir / 'final_reconciliation.csv'}")
    print(final["status"].value_counts().to_string())
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WhyUnmatched deterministic reconciliation")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    run(args.data_dir.resolve())


if __name__ == "__main__":
    main()
