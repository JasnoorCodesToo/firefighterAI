"""
Financial Firefighter — STEP 1: Dataset loading & analysis
=============================================================
Loads the synthetic Razorpay-style transaction dataset and computes
the baseline health metrics we need before we can detect "fires"
(sudden payment incidents) in Step 2.

Run:
    python3 analyze.py
"""

import pandas as pd

CSV_PATH = "/mnt/user-data/uploads/financial_firefighter_transactions.csv"


def load_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Load the transaction CSV and do basic type cleanup."""
    df = pd.read_csv(csv_path)

    # Parse timestamp into a real datetime so we can bucket into time windows later
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Convenience boolean column: True if the transaction succeeded
    df["is_success"] = df["status"] == "SUCCESS"

    # Sort chronologically — important for windowing/time-series analysis
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def overall_metrics(df: pd.DataFrame) -> dict:
    """Headline numbers: success rate, GMV, failed count/GMV."""
    total_txns = len(df)
    success_txns = int(df["is_success"].sum())
    failed_txns = total_txns - success_txns

    total_gmv = float(df["amount"].sum())
    failed_gmv = float(df.loc[~df["is_success"], "amount"].sum())

    success_rate = round(100 * success_txns / total_txns, 2) if total_txns else 0.0

    return {
        "total_transactions": total_txns,
        "success_transactions": success_txns,
        "failed_transactions": failed_txns,
        "success_rate_pct": success_rate,
        "total_gmv": round(total_gmv, 2),
        "failed_gmv": round(failed_gmv, 2),
    }


def success_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Generic breakdown: success rate + volume + failed GMV grouped by a column."""
    grouped = df.groupby(column).agg(
        total_txns=("transaction_id", "count"),
        success_txns=("is_success", "sum"),
        failed_gmv=("amount", lambda s: s[~df.loc[s.index, "is_success"]].sum()),
    )
    grouped["success_rate_pct"] = round(100 * grouped["success_txns"] / grouped["total_txns"], 2)
    grouped["failed_txns"] = grouped["total_txns"] - grouped["success_txns"]
    return grouped.sort_values("success_rate_pct")


def success_rate_over_time(df: pd.DataFrame, freq: str = "15min") -> pd.Series:
    """Resample success rate over fixed time buckets (default 15-minute windows)."""
    s = df.set_index("timestamp")["is_success"].resample(freq).mean() * 100
    return s.round(2)


def print_report(df: pd.DataFrame) -> None:
    metrics = overall_metrics(df)

    print("=" * 60)
    print("FINANCIAL FIREFIGHTER — STEP 1: DATASET ANALYSIS")
    print("=" * 60)

    print(f"\nDataset window: {df['timestamp'].min()}  →  {df['timestamp'].max()}")
    print(f"Total transactions : {metrics['total_transactions']:,}")
    print(f"Successful          : {metrics['success_transactions']:,}")
    print(f"Failed              : {metrics['failed_transactions']:,}")
    print(f"Overall success rate: {metrics['success_rate_pct']}%")
    print(f"Total GMV           : ₹{metrics['total_gmv']:,.2f}")
    print(f"Failed GMV          : ₹{metrics['failed_gmv']:,.2f}")

    print("\n--- Success rate by payment_method ---")
    print(success_rate_by(df, "payment_method")[["total_txns", "failed_txns", "success_rate_pct"]])

    print("\n--- Success rate by bank ---")
    print(success_rate_by(df, "bank")[["total_txns", "failed_txns", "success_rate_pct"]])

    print("\n--- Success rate over time (15-min windows, last 8 shown) ---")
    ts = success_rate_over_time(df)
    print(ts.tail(8))


if __name__ == "__main__":
    data = load_data()
    print_report(data)
