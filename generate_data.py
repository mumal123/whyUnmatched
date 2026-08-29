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
        bank_statement["amount"] = random.choice([payment["payment_amount"], settlement["settlement_amount"],settlement["settlement_amount"]-random.uniform(0.01,0.3)*settlement["settlement_amount"]])
        bank_statement["time"] = f.date_time_between(
            start_date=payment["time"],
            end_date="now"
        )
        bank_statements.append(bank_statement)
    pd.DataFrame(bank_statements).to_csv("data/bank_statements.csv",index=False)
    return bank_statements

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
        ground_truth.append({
            "payment_id": payment["payment_id"],
            "truth": "MATCH" if bank_statement["amount"] == settlement["settlement_amount"] else "AMOUNT_MISMATCH" if bank_statement["amount"] != settlement["settlement_amount"] else "MISSING_SETTELMENT"
            
            
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
            row["truth"]="MISSING_SETTELMENT"
    return settlements, bank_statements, ground_truth
    

payments=generate_payments(num_transactions)
settlements=generate_settlements(payments)
bank_statements=generate_bankStatements(payments, settlements)
ground_truth=ground_truth(payments, settlements, bank_statements)
settlements, bank_statements, ground_truth=inject_missing_settlements(payments, settlements, bank_statements, ground_truth)
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