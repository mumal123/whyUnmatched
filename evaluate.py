import pandas as pd


def build_ground_truth_pairs():
    bank_raw = pd.read_csv("./data/bank_statements.csv")
    ground_truth = pd.read_csv("./data/ground_truth.csv")

    # canonical bank row per payment = first occurrence (duplicates are
    # appended after the original in generate_data.py, so keep="first"
    # always picks the original, never the duplicate copy)
    canonical = bank_raw.dropna(subset=["payment_id"]).drop_duplicates(
        subset="payment_id", keep="first"
    )
    expected_bank_id = dict(zip(canonical["payment_id"], canonical["bank_statement_id"]))

    pairs = []
    for _, row in ground_truth.iterrows():
        pid = row["payment_id"]
        truth = row["truth"]
        expected_id = None if truth == "MISSING_SETTLEMENT" else expected_bank_id.get(pid)
        pairs.append({
            "payment_id": pid,
            "expected_bank_statement_id": expected_id,
            "truth": truth,
        })

    return pd.DataFrame(pairs)


def build_predicted_pairs():
    final = pd.read_csv("./data/final_reconciliation.csv")
    settlements = pd.read_csv("./data/settlements.csv")

    settlement_to_payment = dict(zip(settlements["settlement_id"], settlements["payment_id"]))

    predicted = []
    for _, row in final.iterrows():
        if pd.notna(row["settlement_id"]):
            pid = settlement_to_payment.get(row["settlement_id"])
            predicted.append({
                "payment_id": pid,
                "predicted_bank_statement_id": row["bank_statement_id"],
                "predicted_status": row["status"],
            })

    return pd.DataFrame(predicted)


def classify_pair(row):
    expected = row["expected_bank_statement_id"]
    predicted = row["predicted_bank_statement_id"]

    if pd.isna(expected) and pd.isna(predicted):
        return "TN", "correctly found no match"
    if pd.isna(expected) and pd.notna(predicted):
        return "FP", "false match — should have been no match"
    if pd.notna(expected) and pd.isna(predicted):
        return "FN", "missed match"
    if expected == predicted:
        return "TP", "correct match"
    return "FP", "wrong bank row matched"


def evaluate():
    gt = build_ground_truth_pairs()
    pred = build_predicted_pairs()

    merged = gt.merge(pred, on="payment_id", how="left")
    merged[["result", "reason"]] = merged.apply(
        lambda row: pd.Series(classify_pair(row)), axis=1
    )

    tp = (merged["result"] == "TP").sum()
    fp = (merged["result"] == "FP").sum()
    fn = (merged["result"] == "FN").sum()
    tn = (merged["result"] == "TN").sum()

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"Precision={precision:.4f}  Recall={recall:.4f}")

    exceptions = merged[merged["result"].isin(["FP", "FN"])][
        ["payment_id", "expected_bank_statement_id", "predicted_bank_statement_id",
         "truth", "predicted_status", "result", "reason"]
    ].rename(columns={
        "expected_bank_statement_id": "expected",
        "predicted_bank_statement_id": "predicted",
    })
    exceptions.to_csv("./data/exception_list.csv", index=False)

    print(f"\n{len(exceptions)} exceptions written to exception_list.csv")
    return merged


if __name__ == "__main__":
    evaluate()