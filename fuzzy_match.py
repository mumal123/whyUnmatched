import pandas as pd
import time


def fuzzy_match(bank_leftover, settlement_leftover, amount_tolerance=60, time_window_days=5, partial_fractions=(0.25, 0.5, 0.75), fraction_tolerance=2):

    start_time = time.perf_counter()

    bank_leftover = bank_leftover.copy()
    settlement_leftover = settlement_leftover.copy()
    bank_leftover["time"] = pd.to_datetime(bank_leftover["time"])
    settlement_leftover["time"] = pd.to_datetime(settlement_leftover["time"])

    matched_bank_ids = set()
    matched_settlement_ids = set()
    results = []

    for _, s in settlement_leftover.iterrows():
        window_start = s["time"]
        window_end = s["time"] + pd.Timedelta(days=time_window_days)

        candidates = bank_leftover[
            (bank_leftover["time"] >= window_start) &
            (bank_leftover["time"] <= window_end) &
            (~bank_leftover["bank_statement_id"].isin(matched_bank_ids))
        ]

        best = None
        best_status = None
        best_diff = None

        for _, b in candidates.iterrows():
            diff = abs(b["amount"] - s["settlement_amount"])

            if diff <= amount_tolerance:
                if best is None or diff < best_diff:
                    best, best_status, best_diff = b, "AMOUNT_MISMATCH", diff
                continue

            for fraction in partial_fractions:
                expected = s["settlement_amount"] * fraction
                if abs(b["amount"] - expected) <= fraction_tolerance:
                    if best is None or best_status != "AMOUNT_MISMATCH":
                        best, best_status, best_diff = b, "PARTIAL_PAYMENT", abs(b["amount"] - expected)
                    break

        if best is not None:
            matched_bank_ids.add(best["bank_statement_id"])
            matched_settlement_ids.add(s["settlement_id"])
            results.append({
                "bank_statement_id": best["bank_statement_id"],
                "settlement_id": s["settlement_id"],
                "bank_amount": best["amount"],
                "settlement_amount": s["settlement_amount"],
                "status": best_status,
            })

    for _, b in bank_leftover.iterrows():
        if b["bank_statement_id"] not in matched_bank_ids:
            results.append({
                "bank_statement_id": b["bank_statement_id"],
                "settlement_id": None,
                "bank_amount": b["amount"],
                "settlement_amount": None,
                "status": "UNMATCHED_BANK_CREDIT",
            })

    for _, s in settlement_leftover.iterrows():
        if s["settlement_id"] not in matched_settlement_ids:
            results.append({
                "bank_statement_id": None,
                "settlement_id": s["settlement_id"],
                "bank_amount": None,
                "settlement_amount": s["settlement_amount"],
                "status": "MISSING_SETTLEMENT",
            })

    elapsed = time.perf_counter() - start_time
    return pd.DataFrame(results), elapsed


if __name__ == "__main__":

    exact_result = pd.read_csv("./data/exact_reconciliation.csv")
    bank_statements = pd.read_csv("./data/bank_statements_engine_input.csv")
    settlements = pd.read_csv("./data/settlements.csv")

    bank_statements["amount"] = pd.to_numeric(bank_statements["amount"])
    settlements["settlement_amount"] = pd.to_numeric(settlements["settlement_amount"])

    unmatched_bank_ids = exact_result.loc[exact_result["status"] == "UNMATCHED_BANK_CREDIT", "bank_statement_id"]
    missing_settlement_ids = exact_result.loc[exact_result["status"] == "MISSING_SETTLEMENT", "settlement_id"]

    bank_leftover = bank_statements[bank_statements["bank_statement_id"].isin(unmatched_bank_ids)]
    settlement_leftover = settlements[settlements["settlement_id"].isin(missing_settlement_ids)]

    fuzzy_result, elapsed = fuzzy_match(bank_leftover, settlement_leftover)

    n_records = len(bank_leftover) + len(settlement_leftover)
    print(f"{n_records} leftover records processed in {elapsed:.6f} seconds")
    print(f"Throughput: {n_records / elapsed:.2f} records/sec")
    print(fuzzy_result["status"].value_counts())

    exact_matches_only = exact_result[exact_result["status"] == "MATCH"]

    # payments that never got a settlement at all won't show up anywhere above,
    # since inject_missing_settlements deletes the settlement row entirely —
    # this is the only way to catch that case
    payments = pd.read_csv("./data/payments.csv")
    settled_payment_ids = set(settlements["payment_id"])
    truly_missing = payments[~payments["payment_id"].isin(settled_payment_ids)]

    truly_missing_result = pd.DataFrame({
        "bank_statement_id": None,
        "settlement_id": None,
        "bank_amount": None,
        "settlement_amount": None,
        "status": "MISSING_SETTLEMENT",
    }, index=range(len(truly_missing)))

    final_result = pd.concat([exact_matches_only, fuzzy_result, truly_missing_result], ignore_index=True)
    final_result.to_csv("./data/final_reconciliation.csv", index=False)

    print("\nFinal combined status:")
    print(final_result["status"].value_counts())