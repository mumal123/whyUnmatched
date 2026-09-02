import pandas as pd
import time


def fuzzy_match(
    bank_leftover,
    settlement_leftover,
    amount_match_tolerance=2.0,
    amount_mismatch_tolerance=60.0,
    time_window_days=5,
    partial_fractions=(0.25, 0.5, 0.75),
    fraction_tolerance=2.0,
):
    start_time = time.perf_counter()

    bank_leftover = bank_leftover.copy()
    settlement_leftover = settlement_leftover.copy()

    bank_leftover["time"] = pd.to_datetime(bank_leftover["time"])
    settlement_leftover["time"] = pd.to_datetime(
        settlement_leftover["time"]
    )

    bank_leftover["amount"] = pd.to_numeric(
        bank_leftover["amount"], errors="coerce"
    )

    settlement_leftover["settlement_amount"] = pd.to_numeric(
        settlement_leftover["settlement_amount"], errors="coerce"
    )

    matched_bank_ids = set()
    matched_settlement_ids = set()

    results = []

    for _, s in settlement_leftover.iterrows():

        settlement_id = s["settlement_id"]
        settlement_amount = s["settlement_amount"]
        settlement_time = s["time"]

        window_start = settlement_time
        window_end = settlement_time + pd.Timedelta(
            days=time_window_days
        )

        candidates = bank_leftover[
            (bank_leftover["time"] >= window_start)
            & (bank_leftover["time"] <= window_end)
            & (
                ~bank_leftover["bank_statement_id"].isin(
                    matched_bank_ids
                )
            )
        ].copy()

        if candidates.empty:
            continue

        candidate_records = []

        for _, b in candidates.iterrows():

            bank_id = b["bank_statement_id"]
            bank_amount = b["amount"]

            if pd.isna(bank_amount) or pd.isna(settlement_amount):
                continue

            amount_diff = abs(
                bank_amount - settlement_amount
            )

            if settlement_amount != 0:
                fraction = bank_amount / settlement_amount
            else:
                fraction = None

            status = None
            score = None

            if amount_diff <= amount_match_tolerance:

                status = "MATCH"
                score = amount_diff

            elif amount_diff <= amount_mismatch_tolerance:

                # small absolute drift — always a mismatch, never a
                # partial payment, regardless of settlement size
                status = "AMOUNT_MISMATCH"
                score = amount_diff

            else:

                partial_match = False

                for expected_fraction in partial_fractions:

                    expected_amount = (
                        settlement_amount * expected_fraction
                    )

                    fraction_diff = abs(
                        bank_amount - expected_amount
                    )

                    if fraction_diff <= fraction_tolerance:

                        status = "PARTIAL_PAYMENT"
                        score = fraction_diff
                        partial_match = True
                        break

                # note: a looser "any smaller amount within the window"
                # fallback was removed here — it fired on unrelated
                # bank rows that happened to land in the same window,
                # and stole priority away from the correct AMOUNT_MISMATCH
                # candidate since PARTIAL_PAYMENT ranks above it. The
                # explicit fraction check above already covers every
                # partial payment this dataset actually generates.

            if status is not None:

                candidate_records.append(
                    {
                        "bank_statement_id": bank_id,
                        "bank_amount": bank_amount,
                        "amount_diff": amount_diff,
                        "status": status,
                        "score": score,
                        "bank_time": b["time"],
                    }
                )

        if not candidate_records:
            continue

        status_priority = {
            "MATCH": 0,
            "PARTIAL_PAYMENT": 1,
            "AMOUNT_MISMATCH": 2,
        }

        candidate_records.sort(
            key=lambda x: (
                status_priority[x["status"]],
                x["score"],
                x["bank_time"],
            )
        )

        best = candidate_records[0]

        best_bank_id = best["bank_statement_id"]

        duplicate_candidates = candidate_records[1:]

        matched_bank_ids.add(best_bank_id)
        matched_settlement_ids.add(settlement_id)

        results.append(
            {
                "bank_statement_id": best_bank_id,
                "settlement_id": settlement_id,
                "bank_amount": best["bank_amount"],
                "settlement_amount": settlement_amount,
                "status": best["status"],
            }
        )

        for duplicate in duplicate_candidates:

            duplicate_bank_id = duplicate["bank_statement_id"]

            if duplicate["amount_diff"] <= amount_match_tolerance:

                if duplicate_bank_id not in matched_bank_ids:

                    matched_bank_ids.add(duplicate_bank_id)

                    results.append(
                        {
                            "bank_statement_id": duplicate_bank_id,
                            "settlement_id": settlement_id,
                            "bank_amount": duplicate["bank_amount"],
                            "settlement_amount": settlement_amount,
                            "status": "DUPLICATE",
                        }
                    )

    remaining_bank = bank_leftover[
        ~bank_leftover["bank_statement_id"].isin(
            matched_bank_ids
        )
    ]

    for _, b in remaining_bank.iterrows():

        results.append(
            {
                "bank_statement_id": b["bank_statement_id"],
                "settlement_id": None,
                "bank_amount": b["amount"],
                "settlement_amount": None,
                "status": "UNMATCHED_CREDIT",
            }
        )

    remaining_settlements = settlement_leftover[
        ~settlement_leftover["settlement_id"].isin(
            matched_settlement_ids
        )
    ]

    for _, s in remaining_settlements.iterrows():

        results.append(
            {
                "bank_statement_id": None,
                "settlement_id": s["settlement_id"],
                "bank_amount": None,
                "settlement_amount": s["settlement_amount"],
                "status": "MISSING_BANK_CREDIT",
            }
        )

    result_df = pd.DataFrame(
        results,
        columns=[
            "bank_statement_id",
            "settlement_id",
            "bank_amount",
            "settlement_amount",
            "status",
        ],
    )

    elapsed = time.perf_counter() - start_time

    return result_df, elapsed


if __name__ == "__main__":

    exact_result = pd.read_csv(
        "./data/exact_reconciliation.csv"
    )

    bank_statements = pd.read_csv(
        "./data/bank_statements_engine_input.csv"
    )

    settlements = pd.read_csv(
        "./data/settlements.csv"
    )

    payments = pd.read_csv(
        "./data/payments.csv"
    )

    bank_statements["amount"] = pd.to_numeric(
        bank_statements["amount"],
        errors="coerce",
    )

    settlements["settlement_amount"] = pd.to_numeric(
        settlements["settlement_amount"],
        errors="coerce",
    )

    unmatched_bank_ids = exact_result.loc[
        exact_result["status"] == "UNMATCHED_BANK_CREDIT",
        "bank_statement_id",
    ].dropna()

    missing_bank_credit_settlement_ids = exact_result.loc[
        exact_result["status"] == "MISSING_SETTLEMENT",
        "settlement_id",
    ].dropna()

    bank_leftover = bank_statements[
        bank_statements["bank_statement_id"].isin(
            unmatched_bank_ids
        )
    ].copy()

    settlement_leftover = settlements[
        settlements["settlement_id"].isin(
            missing_bank_credit_settlement_ids
        )
    ].copy()

    fuzzy_result, elapsed = fuzzy_match(
        bank_leftover,
        settlement_leftover,
        amount_match_tolerance=2.0,
        amount_mismatch_tolerance=60.0,
        time_window_days=5,
        partial_fractions=(0.25, 0.5, 0.75),
        fraction_tolerance=2.0,
    )

    n_records = (
        len(bank_leftover)
        + len(settlement_leftover)
    )

    print(
        f"{n_records} leftover records processed "
        f"in {elapsed:.6f} seconds"
    )

    if elapsed > 0:
        print(
            f"Throughput: "
            f"{n_records / elapsed:.2f} records/sec"
        )

    print("\nFuzzy result:")
    if not fuzzy_result.empty:
        print(
            fuzzy_result["status"].value_counts()
        )
    else:
        print("No fuzzy matches generated.")

    exact_matches_only = exact_result[
        exact_result["status"] == "MATCH"
    ].copy()

    # ---------------------------------------------------------
    # DUPLICATE detection
    #
    # A duplicate isn't two candidates racing for the same open
    # settlement — it's an extra bank credit for a settlement whose
    # real bank credit already arrived. That "already arrived" set
    # spans BOTH stages: exact MATCH, and fuzzy MATCH / AMOUNT_MISMATCH
    # / PARTIAL_PAYMENT (all of which mean the settlement is accounted
    # for, just not by this leftover credit).
    # ---------------------------------------------------------

    accounted_for_statuses = {"MATCH", "AMOUNT_MISMATCH", "PARTIAL_PAYMENT"}

    accounted_settlements = pd.concat(
        [
            exact_matches_only[["settlement_amount"]],
            fuzzy_result[
                fuzzy_result["status"].isin(accounted_for_statuses)
            ][["settlement_amount"]],
        ],
        ignore_index=True,
    ).dropna()

    def reclassify_unmatched_credit(row):
        if row["status"] != "UNMATCHED_CREDIT":
            return row["status"]
        close_amount = (
            accounted_settlements["settlement_amount"] - row["bank_amount"]
        ).abs() <= 2.0
        return "DUPLICATE" if close_amount.any() else "UNMATCHED_CREDIT"

    fuzzy_result["status"] = fuzzy_result.apply(
        reclassify_unmatched_credit, axis=1
    )

    settled_payment_ids = set(
        settlements["payment_id"]
    )

    truly_missing = payments[
        ~payments["payment_id"].isin(
            settled_payment_ids
        )
    ].copy()

    truly_missing_result = pd.DataFrame(
        {
            "bank_statement_id": [None] * len(truly_missing),
            "settlement_id": [None] * len(truly_missing),
            "bank_amount": [None] * len(truly_missing),
            "settlement_amount": [None] * len(truly_missing),
            "status": ["MISSING_SETTLEMENT"] * len(truly_missing),
            "payment_id": truly_missing["payment_id"].tolist(),
        }
    )

    final_result = pd.concat(
        [
            exact_matches_only,
            fuzzy_result,
            truly_missing_result,
        ],
        ignore_index=True,
    )

    final_result.to_csv(
        "./data/final_reconciliation.csv",
        index=False,
    )

    print("\nFinal combined status:")
    print(
        final_result["status"].value_counts()
    )

    print("\nCoverage checks:")

    print(
        "Rows in final result:",
        len(final_result)
    )

    print(
        "Rows with missing status:",
        final_result["status"].isna().sum()
    )

    assert final_result["status"].notna().all(), (
        "ERROR: Some records have no status!"
    )

    allowed_statuses = {
        "MATCH",
        "MISSING_SETTLEMENT",
        "AMOUNT_MISMATCH",
        "DUPLICATE",
        "UNMATCHED_CREDIT",
        "PARTIAL_PAYMENT",
        "MISSING_BANK_CREDIT",
    }

    actual_statuses = set(
        final_result["status"].dropna().unique()
    )

    unexpected_statuses = (
        actual_statuses - allowed_statuses
    )

    assert not unexpected_statuses, (
        f"ERROR: Unexpected statuses found: "
        f"{unexpected_statuses}"
    )

    print("Coverage checks passed.")