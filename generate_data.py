from datetime import timedelta

from faker import Faker
import random
import pandas as pd

f=Faker(["en_IN"])
num_transactions = 1000
random.seed(42)
Faker.seed(42)

def generate_payments(num_transactions):
    payments=[]
    for _ in range(num_transactions):
        payment={}
        payment["payment_id"] = f.uuid4()
        payment["order_id"] = f.uuid4()
        payment["payment_amount"] = f.random_int(min=10, max=10000000)
        payment['time'] = f.date_time_this_year()
        payments.append(payment)

    pd.DataFrame(payments).to_csv("data/payments.csv",index=False) 
    return payments

def generate_settlements(payments):
    settlements=[]
    for payment in payments:
        settlement={}
        fee_rate=random.uniform(0.01,0.03)
        settlement["settlement_id"] = f.uuid4()
        settlement["payment_id"] = payment["payment_id"]
        settlement["settlement_amount"] = payment["payment_amount"] -fee_rate*payment["payment_amount"]
        settlement["time"] = f.date_time_between(
            start_date=payment["time"],
            end_date="now"
        )
        settlements.append(settlement)

    pd.DataFrame(settlements).to_csv("data/settlements.csv",index=False)
    return settlements

def generate_bankStatements(payments,settlements):
    bank_statements=[]
    for payment in payments:
        settlement = next(
            s for s in settlements
            if s["payment_id"] == payment["payment_id"]
        )
        bank_statement={}
        bank_statement["bank_statement_id"] = f.uuid4()
        bank_statement["payment_id"] = payment["payment_id"]
        bank_statement["amount"] =  settlement["settlement_amount"]
        bank_statement["time"] = f.date_time_between(
            start_date=settlement["time"],
            end_date=settlement["time"] + timedelta(days=2)
        )
        bank_statements.append(bank_statement)
    pd.DataFrame(bank_statements).to_csv("data/bank_statements.csv",index=False)
    return bank_statements

def generate_bankStatement_input(bank_statements):
    return[
        {k: v for k, v in b.items() if k != "payment_id"} for b in bank_statements
    ]

def ground_truth(payments, settlements, bank_statements):
    ground_truth=[]
    for payment in payments:
        settlement = next(
            s for s in settlements
            if s["payment_id"] == payment["payment_id"]
        )
        bank_statement = next(
            b for b in bank_statements
            if b["payment_id"] == payment["payment_id"]
        )
        if settlement is None:
            truth = "MISSING_SETTLEMENT"
        elif bank_statement is None:
            truth = "UNMATCHED_BANK_CREDIT"  # settlement exists but no bank credit landed yet
        elif bank_statement["amount"] == settlement["settlement_amount"]:
            truth = "MATCH"
        else:
            truth = "AMOUNT_MISMATCH"
        ground_truth.append({
            "payment_id": payment["payment_id"],
            "truth": truth
        })
    pd.DataFrame(ground_truth).to_csv("data/ground_truth.csv",index=False)
    return ground_truth

def inject_missing_settlements(payments, settlements, bank_statements, ground_truth):
    missing_count=int(0.05*len(payments))
    missing_payments=random.sample(payments,missing_count)
    missing_payment_ids=[payment["payment_id"] for payment in missing_payments]
    settlements=[settlement for settlement in settlements if settlement["payment_id"] not in missing_payment_ids]
    bank_statements=[bank_statement for bank_statement in bank_statements if bank_statement["payment_id"] not in missing_payment_ids]
    for row in ground_truth:
        if row["payment_id"] in missing_payment_ids:
            row["truth"]="MISSING_SETTLEMENT"
    return settlements, bank_statements, ground_truth

def inject_partial_payments(payments, bank_statements, ground_truth, settlements):
    settlement_by_payment = {s["payment_id"]: s for s in settlements}
    eligible_payments = [
        p for p in payments
        if p["payment_id"] in settlement_by_payment
    ]
    partial_count = int(0.05 * len(payments))
    partial_payments = random.sample(eligible_payments,  min(partial_count, len(eligible_payments)))
    partial_payment_ids = {p["payment_id"] for p in partial_payments}
 
    for b in bank_statements:
        if b["payment_id"] in partial_payment_ids:
            settlement = settlement_by_payment.get(b["payment_id"])
        
            paid_fraction = random.choice([0.25, 0.5, 0.75])
            b["amount"] = round(settlement["settlement_amount"] * paid_fraction, 2)
 
    for row in ground_truth:
        if row["payment_id"] in partial_payment_ids:
            row["truth"] = "PARTIAL_PAYMENT"
 
    return bank_statements, ground_truth

def inject_amount_mismatches(payments, bank_statements, ground_truth, settlements):
    settlement_by_payment = {s["payment_id"]: s for s in settlements}
    truth_by_payment = {row["payment_id"]: row["truth"] for row in ground_truth}

    eligible_payments = [
        p for p in payments
        if p["payment_id"] in settlement_by_payment
        and truth_by_payment[p["payment_id"]] == "MATCH"   # only touch untouched, clean records
    ]

    mismatch_count = int(0.05 * len(payments))
    mismatch_payments = random.sample(
        eligible_payments, min(mismatch_count, len(eligible_payments))
    )
    mismatch_ids = {p["payment_id"] for p in mismatch_payments}

    for b in bank_statements:
        if b["payment_id"] in mismatch_ids:
            settlement = settlement_by_payment[b["payment_id"]]
            # small drift: an extra bank charge or rounding difference, not a big chunk missing
            drift = round(random.uniform(1, 50), 2)
            b["amount"] = round(settlement["settlement_amount"] - drift, 2)

    for row in ground_truth:
        if row["payment_id"] in mismatch_ids:
            row["truth"] = "AMOUNT_MISMATCH"

    return bank_statements, ground_truth

def inject_duplicate_payments(payments, bank_statements, ground_truth):
    bank_by_payment = {
        b["payment_id"]: b
        for b in bank_statements
        if b["payment_id"] is not None
    }

    truth_by_payment = {
        row["payment_id"]: row["truth"]
        for row in ground_truth
    }

    eligible_payments = [
        p for p in payments
        if p["payment_id"] in bank_by_payment
        and truth_by_payment[p["payment_id"]] not in {
            "MISSING_SETTLEMENT",
            "PARTIAL_PAYMENT",
            "AMOUNT_MISMATCH",
        }
    ]

    dup_count = int(0.03 * len(payments))

    dup_source_payments = random.sample(
        eligible_payments,
        min(dup_count, len(eligible_payments))
    )

    duplicate_ids = {
        p["payment_id"]
        for p in dup_source_payments
    }

    new_duplicate_rows = []

    for payment in dup_source_payments:
        original = bank_by_payment[payment["payment_id"]]

        duplicate = dict(original)

        duplicate["bank_statement_id"] = f.uuid4()
        duplicate["time"] = original["time"]

        new_duplicate_rows.append(duplicate)

    bank_statements.extend(new_duplicate_rows)

    for row in ground_truth:
        if row["payment_id"] in duplicate_ids:
            row["truth"] = "DUPLICATE_PAYMENT"

    return bank_statements, ground_truth

def inject_unmatched_bank_credits(bank_statements, count=15):
    # bank credits with no matching payment_id at all — e.g. stray refund reversals, manual transfers
    stray_rows = []
    for _ in range(count):
        stray_rows.append({
            "bank_statement_id": f.uuid4(),
            "payment_id": None,
            "amount": f.random_int(min=100, max=50000),
            "time": f.date_time_this_year(),
        })
    return bank_statements + stray_rows
    

payments=generate_payments(num_transactions)
settlements=generate_settlements(payments)
bank_statements=generate_bankStatements(payments, settlements)

ground_truth=ground_truth(payments, settlements, bank_statements)
settlements, bank_statements, ground_truth=inject_missing_settlements(payments, settlements, bank_statements, ground_truth)
bank_statements, ground_truth = inject_partial_payments(
    payments, bank_statements, ground_truth, settlements
)
bank_statements, ground_truth = inject_amount_mismatches(
    payments, bank_statements, ground_truth, settlements
)
bank_statements, ground_truth = inject_duplicate_payments(
    payments, bank_statements, ground_truth
)
bank_statements = inject_unmatched_bank_credits(bank_statements, count=15)
bank_statements_input=generate_bankStatement_input(bank_statements)
pd.DataFrame(bank_statements_input).to_csv("data/bank_statements_engine_input.csv", index=False)
pd.DataFrame(settlements).to_csv(
    "data/settlements.csv",
    index=False
)

pd.DataFrame(bank_statements).to_csv(
    "data/bank_statements.csv",
    index=False
)

pd.DataFrame(ground_truth).to_csv(
    "data/ground_truth.csv",
    index=False
)

print(pd.DataFrame(ground_truth)["truth"].value_counts())
print(len(pd.read_csv("data/payments.csv")))
print(len(pd.read_csv("data/settlements.csv")))
print(len(pd.read_csv("data/bank_statements.csv")))
print(len(pd.read_csv("data/ground_truth.csv")))
