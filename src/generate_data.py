"""
Dataset Generator: E-Commerce Behavioral & Revenue Dataset
Generates a realistic synthetic dataset for EDA purposes.
"""
import numpy as np
import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()
np.random.seed(42)
random.seed(42)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
N_CUSTOMERS = 8000
N_TRANSACTIONS = 75000
START_DATE = datetime(2022, 1, 1)
END_DATE   = datetime(2024, 12, 31)

CATEGORIES = {
    "Electronics":   {"avg_price": 320, "std": 180, "margin": 0.18},
    "Apparel":       {"avg_price": 65,  "std": 45,  "margin": 0.52},
    "Home & Garden": {"avg_price": 95,  "std": 60,  "margin": 0.41},
    "Sports":        {"avg_price": 110, "std": 70,  "margin": 0.38},
    "Beauty":        {"avg_price": 42,  "std": 28,  "margin": 0.60},
    "Books":         {"avg_price": 18,  "std": 12,  "margin": 0.30},
    "Toys":          {"avg_price": 35,  "std": 22,  "margin": 0.44},
    "Food & Grocery":{"avg_price": 28,  "std": 15,  "margin": 0.22},
}

CHANNELS = ["Organic Search", "Paid Search", "Social Media", "Email", "Direct", "Referral", "Display Ad"]
DEVICES   = ["Mobile", "Desktop", "Tablet"]
REGIONS   = ["West", "Midwest", "South", "Northeast", "Southwest"]
SEGMENTS  = ["New", "Occasional", "Regular", "Loyal", "Champion"]

# ─── CUSTOMERS ────────────────────────────────────────────────────────────────
def generate_customers(n):
    rows = []
    for cid in range(1, n + 1):
        seg = np.random.choice(SEGMENTS, p=[0.30, 0.25, 0.22, 0.15, 0.08])
        # age skews differently per segment
        base_age = {"New": 28, "Occasional": 32, "Regular": 35, "Loyal": 38, "Champion": 41}[seg]
        age = max(18, int(np.random.normal(base_age, 8)))
        join_year = np.random.choice([2021, 2022, 2023, 2024],
                                     p=[0.15, 0.30, 0.35, 0.20])
        rows.append({
            "customer_id": f"C{cid:05d}",
            "age": age,
            "gender": np.random.choice(["Male", "Female", "Non-binary/Other"], p=[0.46, 0.49, 0.05]),
            "region": np.random.choice(REGIONS),
            "segment": seg,
            "acquisition_channel": np.random.choice(CHANNELS),
            "join_year": join_year,
            "loyalty_points": max(0, int(np.random.exponential(500) if seg in ["Loyal","Champion"] else np.random.exponential(80))),
            "email_opt_in": np.random.choice([True, False], p=[0.72, 0.28]),
        })
    return pd.DataFrame(rows)

# ─── TRANSACTIONS ─────────────────────────────────────────────────────────────
def generate_transactions(n, customers_df):
    segment_probs = customers_df.groupby("segment").size() / len(customers_df)
    
    # Loyal/Champion customers generate more transactions
    weights = customers_df["segment"].map({
        "New": 0.5, "Occasional": 1.0, "Regular": 2.0, "Loyal": 4.0, "Champion": 7.0
    })
    weights /= weights.sum()
    cids = np.random.choice(customers_df["customer_id"], size=n, p=weights)

    rows = []
    total_days = (END_DATE - START_DATE).days

    for i, cid in enumerate(cids):
        cat  = np.random.choice(list(CATEGORIES.keys()))
        cfg  = CATEGORIES[cat]
        price = max(1.0, round(np.random.normal(cfg["avg_price"], cfg["std"]), 2))
        qty   = np.random.choice([1,1,1,2,2,3,4], p=[0.45,0.20,0.10,0.10,0.07,0.05,0.03])

        # Simulate seasonality: Q4 gets ~40% more traffic
        day_offset = int(np.random.beta(1.5, 1.2) * total_days)
        txn_date   = START_DATE + timedelta(days=day_offset)
        # Q4 boost
        if txn_date.month in [11, 12]:
            if random.random() < 0.3:
                txn_date = txn_date.replace(month=random.choice([11,12]),
                                             day=random.randint(1,28))

        discount   = round(np.random.choice([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                                             p=[0.45,0.15,0.15,0.10,0.08,0.04,0.03]), 2)
        gross      = round(price * qty, 2)
        net        = round(gross * (1 - discount), 2)
        margin     = round(net * cfg["margin"], 2)

        is_returned = False
        if random.random() < 0.08:  # 8% return rate
            is_returned = True

        rows.append({
            "transaction_id":  f"T{i+1:07d}",
            "customer_id":     cid,
            "date":            txn_date.strftime("%Y-%m-%d"),
            "category":        cat,
            "product_price":   price,
            "quantity":        qty,
            "discount_pct":    discount,
            "gross_revenue":   gross,
            "net_revenue":     net,
            "gross_margin":    margin,
            "channel":         np.random.choice(CHANNELS),
            "device":          np.random.choice(DEVICES, p=[0.58, 0.34, 0.08]),
            "is_returned":     is_returned,
            "session_minutes": max(1, round(np.random.exponential(8), 1)),
            "pages_viewed":    np.random.randint(1, 18),
        })

    return pd.DataFrame(rows)

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating customers...")
    customers = generate_customers(N_CUSTOMERS)
    customers.to_csv("data/customers.csv", index=False)
    print(f"  → {len(customers):,} customers saved.")

    print("Generating transactions...")
    transactions = generate_transactions(N_TRANSACTIONS, customers)
    transactions.to_csv("data/transactions.csv", index=False)
    print(f"  → {len(transactions):,} transactions saved.")

    print("Done!")
