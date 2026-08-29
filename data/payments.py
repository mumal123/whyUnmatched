from faker import Faker
import random
import pandas as pd

f=Faker(["en_IN"])
num_transactions = 1000
random.seed(42)
Faker.seed(42)


payments=[]
for _ in range(num_transactions):
    payment={}
    payment["payment_id"] = f.uuid4()
    payment["order_id"] = f.uuid4()
    payment["payment_amount"] = f.random_int(min=10, max=10000000)
    payment['time'] = f.date_time_this_year()
    payments.append(payment)

pd.DataFrame(payments).to_csv("payments.csv",index=False)