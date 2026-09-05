import pandas as pd
import time


def exact_match(bank_statements, settlements, settlement_delay_days=2):
    start_time = time.perf_counter()

    required_bank = {"bank_statement_id", "amount", "time"}
    required_settlement = {"settlement_id", "settlement_amount", "time"}
    missing_bank = required_bank - set(bank_statements.columns)
    missing_settlement = required_settlement - set(settlements.columns)
    if missing_bank or missing_settlement:
        problems = []
        if missing_bank:
            problems.append(f"bank statements missing {sorted(missing_bank)}")
        if missing_settlement:
            problems.append(f"settlements missing {sorted(missing_settlement)}")
        raise ValueError("; ".join(problems))

    bank_statements = bank_statements.copy()
    settlements = settlements.copy()
    bank_statements["time"] = pd.to_datetime(bank_statements["time"])
    settlements["time"] = pd.to_datetime(settlements["time"])
    bank_statements["amount"] = pd.to_numeric(bank_statements["amount"], errors="coerce")
    settlements["settlement_amount"] = pd.to_numeric(
        settlements["settlement_amount"], errors="coerce"
    )

    # explicit tie-break: preserves the file's original row order (the
    # original bank credit always appears before any duplicate copy),
    # so ties on identical timestamps resolve deterministically instead
    # of depending on merge/sort implementation details
    bank_statements["_row_order"] = range(len(bank_statements))

    candidates = bank_statements.merge(
        settlements,
        left_on="amount", right_on="settlement_amount",
        suffixes=("_bank", "_settlement")
    )

    window_end = candidates["time_settlement"] + pd.Timedelta(days=settlement_delay_days)
    in_window = (candidates["time_bank"] >= candidates["time_settlement"]) & \
                (candidates["time_bank"] <= window_end)
    candidates = candidates[in_window]

    # Resolve candidates in one deterministic pass. Chaining two independent
    # drop_duplicates calls can discard a settlement's valid second choice
    # after its first choice was already claimed by another settlement.
    candidates = candidates.sort_values(
        ["time_bank", "time_settlement", "_row_order", "settlement_id"],
        kind="stable",
    )
    selected_rows = []
    used_bank_ids = set()
    used_settlement_ids = set()
    for _, candidate in candidates.iterrows():
        bank_id = candidate["bank_statement_id"]
        settlement_id = candidate["settlement_id"]
        if bank_id in used_bank_ids or settlement_id in used_settlement_ids:
            continue
        selected_rows.append(candidate)
        used_bank_ids.add(bank_id)
        used_settlement_ids.add(settlement_id)

    candidates = pd.DataFrame(selected_rows, columns=candidates.columns)

    matched_bank_ids = set(candidates["bank_statement_id"])
    matched_settlement_ids = set(candidates["settlement_id"])

    result_rows = []
    result_rows.append(candidates.assign(status="MATCH")[
        ["bank_statement_id", "settlement_id", "amount", "settlement_amount", "status"]
    ].rename(columns={"amount": "bank_amount"}))

    unmatched_bank = bank_statements[~bank_statements["bank_statement_id"].isin(matched_bank_ids)]
    result_rows.append(pd.DataFrame({
        "bank_statement_id": unmatched_bank["bank_statement_id"],
        "settlement_id": None,
        "bank_amount": unmatched_bank["amount"],
        "settlement_amount": None,
        "status": "UNMATCHED_BANK_CREDIT",
    }))

    missing_settlement = settlements[~settlements["settlement_id"].isin(matched_settlement_ids)]
    result_rows.append(pd.DataFrame({
        "bank_statement_id": None,
        "settlement_id": missing_settlement["settlement_id"],
        "bank_amount": None,
        "settlement_amount": missing_settlement["settlement_amount"],
        "status": "MISSING_SETTLEMENT",
    }))

    result_rows = [df for df in result_rows if not df.empty]
    result_df = pd.concat(result_rows, ignore_index=True)
    elapsed = time.perf_counter() - start_time
    return result_df, elapsed


if __name__ == "__main__":
    bank_statements = pd.read_csv("../data/bank_statements_engine_input.csv")
    settlements = pd.read_csv("../data/settlements.csv")
    bank_statements["amount"] = pd.to_numeric(bank_statements["amount"])
    settlements["settlement_amount"] = pd.to_numeric(settlements["settlement_amount"])

    result, elapsed = exact_match(bank_statements, settlements)
    n_records = len(bank_statements) + len(settlements)
    print(f"{n_records} records processed in {elapsed:.6f} seconds")
    print(f"Throughput: {n_records / elapsed:.2f} records/sec")
    print(result["status"].value_counts())

    result.to_csv("../data/exact_reconciliation.csv", index=False)
