"""
Fetch Razorpay Data — WhyUnmatched
===================================
Pulls real payments and settlements from Razorpay's Test Mode API and
normalizes them into the exact CSV schema exact_match.py / fuzzy_match.py
already expect — so the existing reconciliation engine runs unchanged
against real Razorpay data, not just the synthetic generator.

Test Mode is free, uses fake card numbers, and requires no KYC.

Setup:
    1. Get test keys: Razorpay Dashboard -> Settings -> API Keys -> Generate Test Key
    2. Add to your .env file:
         RAZORPAY_KEY_ID=rzp_test_xxxxx
         RAZORPAY_KEY_SECRET=xxxxx

Usage:
    python fetch_razorpay_data.py
    python fetch_razorpay_data.py --count 50
"""

import argparse
import base64
import os

import pandas as pd
import requests

from dotenv import load_dotenv
load_dotenv()

RAZORPAY_BASE_URL = "https://api.razorpay.com/v1"


def _auth_header(key_id: str, key_secret: str) -> dict:
    token = base64.b64encode(f"{key_id}:{key_secret}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def fetch_payments(key_id: str, key_secret: str, count: int = 100) -> list[dict]:
    """
    GET /v1/payments — Razorpay paginates in batches of up to 100.
    Returns raw payment dicts as Razorpay's API returns them.
    """
    headers = _auth_header(key_id, key_secret)
    all_payments = []
    skip = 0
    batch_size = min(count, 100)

    while len(all_payments) < count:
        resp = requests.get(
            f"{RAZORPAY_BASE_URL}/payments",
            headers=headers,
            params={"count": batch_size, "skip": skip},
            timeout=15,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if not items:
            break
        all_payments.extend(items)
        skip += len(items)
        if len(items) < batch_size:
            break  # no more pages

    return all_payments[:count]


def fetch_settlements(key_id: str, key_secret: str, count: int = 100) -> list[dict]:
    """GET /v1/settlements — same pagination pattern as payments."""
    headers = _auth_header(key_id, key_secret)
    all_settlements = []
    skip = 0
    batch_size = min(count, 100)

    while len(all_settlements) < count:
        resp = requests.get(
            f"{RAZORPAY_BASE_URL}/settlements",
            headers=headers,
            params={"count": batch_size, "skip": skip},
            timeout=15,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if not items:
            break
        all_settlements.extend(items)
        skip += len(items)
        if len(items) < batch_size:
            break

    return all_settlements[:count]


def normalize_payments(raw_payments: list[dict]) -> pd.DataFrame:
    """
    Razorpay payment object -> our schema:
      payment_id, order_id, payment_amount, time
    Razorpay amounts are in paise (integer); payment_amount here is kept in
    paise too, matching the synthetic generator's raw-integer convention —
    if you want rupees, divide by 100 before writing the CSV.
    """
    rows = []
    for p in raw_payments:
        rows.append({
            "payment_id": p["id"],
            "order_id": p.get("order_id") or "",
            "payment_amount": p["amount"],
            "time": pd.to_datetime(p["created_at"], unit="s"),
        })
    return pd.DataFrame(rows)


def normalize_settlements(raw_settlements: list[dict]) -> pd.DataFrame:
    """
    Razorpay settlement object -> our schema:
      settlement_id, payment_id, settlement_amount, time

    IMPORTANT CAVEAT: a Razorpay settlement is a BATCH payout covering many
    payments in one bank transfer — it does not carry a single payment_id.
    Real settlement-to-payment mapping requires the Settlement Reconciliation
    (recon) report or the Transactions API, not the plain /settlements
    endpoint. Since our matching engine assumes one settlement per payment
    (a documented MVP simplification — see README), payment_id is left
    blank here; these rows are usable for testing ingestion/schema shape,
    not for a real precision/recall run against ground truth.
    """
    rows = []
    for s in raw_settlements:
        rows.append({
            "settlement_id": s["id"],
            "payment_id": "",  # see caveat above — needs recon report to fill in
            "settlement_amount": s["amount"],
            "time": pd.to_datetime(s["created_at"], unit="s"),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Fetch real Razorpay Test Mode data")
    parser.add_argument("--count", type=int, default=50, help="Max records to fetch per endpoint (default: 50)")
    parser.add_argument("--out-dir", type=str, default="data_razorpay_live", help="Output directory for CSVs")
    args = parser.parse_args()

    key_id = os.environ.get("RAZORPAY_KEY_ID")
    key_secret = os.environ.get("RAZORPAY_KEY_SECRET")

    if not key_id or not key_secret:
        print("Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in your .env file.")
        print("Get test keys: Razorpay Dashboard -> Settings -> API Keys -> Generate Test Key")
        exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Fetching up to {args.count} payments...")
    raw_payments = fetch_payments(key_id, key_secret, args.count)
    print(f"  got {len(raw_payments)} payments")

    print(f"Fetching up to {args.count} settlements...")
    raw_settlements = fetch_settlements(key_id, key_secret, args.count)
    print(f"  got {len(raw_settlements)} settlements")

    if not raw_payments and not raw_settlements:
        print("\nNo data returned. Test Mode accounts start empty — you need to generate")
        print("test transactions first. Easiest ways:")
        print("  1. Razorpay Dashboard -> Payments -> Create test payment link, pay it")
        print("     with a test card (4111 1111 1111 1111, any future expiry, any CVV)")
        print("  2. Or use the Orders API to create+capture a payment programmatically")
        return

    payments_df = normalize_payments(raw_payments)
    settlements_df = normalize_settlements(raw_settlements)

    payments_path = os.path.join(args.out_dir, "payments.csv")
    settlements_path = os.path.join(args.out_dir, "settlements.csv")
    payments_df.to_csv(payments_path, index=False)
    settlements_df.to_csv(settlements_path, index=False)

    print(f"\nWrote {len(payments_df)} payments to {payments_path}")
    print(f"Wrote {len(settlements_df)} settlements to {settlements_path}")
    print("\nNote: settlement.payment_id is blank (see docstring caveat) — the")
    print("plain /settlements endpoint doesn't expose per-payment breakdown.")
    print("This proves real API ingestion works; for a full end-to-end real-data")
    print("run you'd additionally need the Settlement Recon report.")


if __name__ == "__main__":
    main()