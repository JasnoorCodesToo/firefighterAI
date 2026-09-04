"""
Financial Firefighter — STEP 2: FIRE DETECTOR
=================================================
Detects sudden payment-success-rate incidents ("fires") in a stream of
transactions, WITHOUT any hardcoded knowledge of which bank / payment
method / etc. is expected to be at fault. It works purely off statistics:

  1. Bucket transactions into fixed time windows (default 15 minutes).
  2. Treat the most recent window as "current" and everything before it
     (minus a short buffer, to avoid mixing in the incident itself) as
     the historical "baseline".
  3. Compare current vs. baseline success rate using both:
       - a simple relative % drop threshold, and
       - a z-score against the natural variance of past windows,
     so a single noisy window doesn't trigger a false alarm.
  4. If an incident is confirmed, drill into payment_method / bank /
     device / city / failure_reason to find which specific segment(s)
     degraded the most (the "seat of the fire"), again by comparing
     each segment's current vs. baseline success rate — not by looking
     for a specific known-bad value.

Run:
    python3 detector.py
"""

import pandas as pd
import numpy as np

from analyze import load_data, CSV_PATH

# ---------------------------------------------------------------------------
# Config / tunable thresholds — these are the "sensitivity knobs" of the
# smoke detector. Kept as constants (not hardcoded business values) so they
# are easy to tune per merchant / traffic volume.
# ---------------------------------------------------------------------------
DEFAULT_WINDOW = "15min"      # size of each time bucket
MIN_CURRENT_TXNS = 30         # ignore windows too small to be statistically meaningful
DROP_PCT_THRESHOLD = 15.0     # relative % drop in success rate to call it a "fire"
Z_SCORE_THRESHOLD = 2.0       # how many baseline std-devs below normal counts as anomalous
MIN_BASELINE_WINDOWS = 4      # need at least this many prior windows to trust the baseline
MIN_SEGMENT_TXNS = 5          # min transactions in a segment (e.g. a bank) to consider it a suspect

SEGMENT_DIMENSIONS = ["payment_method", "bank", "device", "city"]


# ---------------------------------------------------------------------------
# 1. Windowing
# ---------------------------------------------------------------------------
def build_time_windows(df: pd.DataFrame, freq: str = DEFAULT_WINDOW) -> pd.DataFrame:
    """
    Bucket the raw transaction log into fixed-size time windows and compute
    a success-rate + GMV summary for each window.
    """
    def _summarize_group(g: pd.DataFrame) -> pd.Series:
        total = len(g)
        success = int(g["is_success"].sum())
        failed_mask = ~g["is_success"]
        return pd.Series(
            {
                "total_txns": total,
                "success_txns": success,
                "gmv": g["amount"].sum(),
                "failed_gmv": g.loc[failed_mask, "amount"].sum(),
            }
        )

    windows = df.set_index("timestamp").resample(freq).apply(_summarize_group)

    # Drop completely empty windows (can happen at the very edges)
    windows = windows[windows["total_txns"] > 0].copy()

    windows["failed_txns"] = windows["total_txns"] - windows["success_txns"]
    windows["success_rate"] = 100 * windows["success_txns"] / windows["total_txns"]

    return windows


# ---------------------------------------------------------------------------
# 2. Current window vs. historical baseline
# ---------------------------------------------------------------------------
def get_current_snapshot(df: pd.DataFrame, freq: str = DEFAULT_WINDOW) -> pd.Series:
    """
    Build the "current" snapshot as a TRAILING window: the last `freq`
    worth of actual transactions, ending at the last timestamp in the
    data. This is deliberately NOT a fixed calendar-grid bucket, because
    the very last grid bucket in any live/batch dataset is often partial
    (cut off mid-window) which would understate a fire in progress.
    """
    max_ts = df["timestamp"].max()
    window_start = max_ts - pd.Timedelta(freq)
    current_df = df[df["timestamp"] > window_start]

    total = len(current_df)
    success = int(current_df["is_success"].sum())
    failed_mask = ~current_df["is_success"]

    return pd.Series(
        {
            "window_start": window_start,
            "window_end": max_ts,
            "total_txns": total,
            "success_txns": success,
            "failed_txns": total - success,
            "success_rate": 100 * success / total if total else 0.0,
            "gmv": current_df["amount"].sum(),
            "failed_gmv": current_df.loc[failed_mask, "amount"].sum(),
        }
    ), current_df, window_start


def split_current_and_baseline(df: pd.DataFrame, freq: str = DEFAULT_WINDOW,
                                min_current_txns: int = MIN_CURRENT_TXNS):
    """
    Build the current trailing-window snapshot, and the baseline set of
    fixed-grid windows computed from everything BEFORE that window (so
    the incident itself never leaks into what we consider "normal").

    Returns (current_snapshot, current_raw_df, baseline_windows_df) or
    (None, None, None) if there isn't enough data.
    """
    if df.empty:
        return None, None, None

    current, current_df, window_start = get_current_snapshot(df, freq)

    if current["total_txns"] < min_current_txns:
        return None, None, None

    baseline_source = df[df["timestamp"] <= window_start]
    baseline_windows = build_time_windows(baseline_source, freq)

    return current, current_df, baseline_windows


def compute_baseline_stats(baseline: pd.DataFrame) -> dict:
    """
    Summarize the baseline: a volume-weighted average success rate (fair
    comparison basis) plus the std-dev of per-window rates (used for the
    z-score anomaly check).
    """
    total_txns = baseline["total_txns"].sum()
    total_success = baseline["success_txns"].sum()
    weighted_rate = 100 * total_success / total_txns if total_txns else 0.0
    rate_std = baseline["success_rate"].std(ddof=0)

    return {
        "baseline_success_rate": round(weighted_rate, 2),
        "baseline_rate_std": round(float(rate_std), 4) if not np.isnan(rate_std) else 0.0,
        "baseline_windows": len(baseline),
    }


# ---------------------------------------------------------------------------
# 3. Incident detection logic
# ---------------------------------------------------------------------------
def classify_severity(drop_pct: float) -> str:
    """Map a relative success-rate drop (%) to a severity label."""
    if drop_pct >= 30:
        return "CRITICAL"
    elif drop_pct >= 20:
        return "HIGH"
    elif drop_pct >= 10:
        return "MEDIUM"
    elif drop_pct >= 5:
        return "LOW"
    return "NONE"


def detect_incident(
    df: pd.DataFrame,
    freq: str = DEFAULT_WINDOW,
    drop_threshold: float = DROP_PCT_THRESHOLD,
    z_threshold: float = Z_SCORE_THRESHOLD,
    min_current_txns: int = MIN_CURRENT_TXNS,
    min_baseline_windows: int = MIN_BASELINE_WINDOWS,
) -> dict:
    """
    Core fire-detection routine. Returns a structured incident dict
    (incident_detected=False if nothing anomalous is found).
    """
    current, current_df, baseline = split_current_and_baseline(df, freq, min_current_txns)

    if current is None or baseline is None or len(baseline) < min_baseline_windows:
        return {
            "incident_detected": False,
            "reason": "Not enough data to establish a reliable baseline yet.",
        }

    base_stats = compute_baseline_stats(baseline)
    baseline_rate = float(base_stats["baseline_success_rate"])
    current_rate = round(float(current["success_rate"]), 2)

    # Relative drop: how much worse is "now" vs. "normal", in percentage terms
    drop_pct = round(100 * (baseline_rate - current_rate) / baseline_rate, 2) if baseline_rate else 0.0
    drop_pct = float(drop_pct)

    # Z-score: how many standard deviations below the normal *window-to-window*
    # fluctuation is the current window? Protects against flagging normal noise.
    rate_std = base_stats["baseline_rate_std"] or 1e-6  # avoid divide-by-zero
    z_score = float(round((baseline_rate - current_rate) / rate_std, 2))

    # Flag as an incident if EITHER signal fires:
    #   - the drop is large in absolute/relative terms (drop_pct), or
    #   - the drop is statistically unusual vs. normal window-to-window
    #     noise (z_score) — this catches incidents that are "diluted"
    #     at the whole-portfolio level (e.g. one bank+method combo on
    #     fire) but are still a significant, non-random deviation.
    # Both must at least agree the rate went DOWN (positive values).
    is_incident = drop_pct > 0 and ((drop_pct >= drop_threshold) or (z_score >= z_threshold))

    result = {
        "incident_detected": bool(is_incident),
        "window_size": freq,
        "current_window_start": str(current["window_start"]),
        "current_window_end": str(current["window_end"]),
        "baseline_success_rate": baseline_rate,
        "current_success_rate": current_rate,
        "drop_percentage": drop_pct,
        "z_score": z_score,
        "affected_transactions": int(current["failed_txns"]),
        "revenue_at_risk": round(float(current["failed_gmv"]), 2),
        "current_window_txns": int(current["total_txns"]),
        "baseline_windows_used": base_stats["baseline_windows"],
    }

    if not is_incident:
        result["severity"] = "NONE"
        return result

    result["severity"] = classify_severity(drop_pct)

    # Root-cause drill-down (Step 2, point 6) — only run when we actually
    # have a confirmed incident, and only look at the current window's data
    # vs. everything that came before it.
    baseline_df = df[df["timestamp"] <= current["window_start"]]
    root_cause = find_root_cause(current_df, baseline_df)
    result.update(root_cause)

    # If a specific segment (e.g. one bank+method combo) is degrading far
    # worse than the overall portfolio number suggests, escalate severity —
    # a fire concentrated in one corner of the building is still a fire,
    # even if the whole building's smoke alarm hasn't tripped yet.
    segment = result.get("segment_analysis") or {}
    segment_drop = segment.get("segment_drop_percentage", 0)
    if segment_drop and segment_drop > drop_pct:
        segment_severity = classify_severity(segment_drop)
        severity_rank = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        if severity_rank[segment_severity] > severity_rank[result["severity"]]:
            result["portfolio_severity"] = result["severity"]
            result["severity"] = segment_severity

    return result


# ---------------------------------------------------------------------------
# 4. Root-cause / concentration analysis
# ---------------------------------------------------------------------------
def worst_offender(current_df: pd.DataFrame, baseline_df: pd.DataFrame, column: str,
                    min_segment_txns: int = MIN_SEGMENT_TXNS):
    """
    For a given dimension (e.g. 'bank'), find the segment value whose
    success rate degraded the MOST vs its own historical baseline.
    This is what lets us find "HDFC" or "UPI" generically — we never
    reference those names in code, we just measure degradation per segment.
    """
    cur_counts = current_df.groupby(column).agg(
        total=("transaction_id", "count"), success=("is_success", "sum")
    )
    cur_counts = cur_counts[cur_counts["total"] >= min_segment_txns]
    if cur_counts.empty:
        return None, None

    cur_counts["current_rate"] = 100 * cur_counts["success"] / cur_counts["total"]

    base_counts = baseline_df.groupby(column).agg(
        total=("transaction_id", "count"), success=("is_success", "sum")
    )
    base_counts["baseline_rate"] = 100 * base_counts["success"] / base_counts["total"]

    merged = cur_counts.join(base_counts[["baseline_rate"]], how="left")
    merged["baseline_rate"] = merged["baseline_rate"].fillna(merged["current_rate"])
    merged["degradation"] = merged["baseline_rate"] - merged["current_rate"]

    if merged.empty or merged["degradation"].max() <= 0:
        # Nothing degraded in this dimension relative to its own history
        top_value = merged["degradation"].idxmax() if not merged.empty else None
        return top_value, merged
    top_value = merged["degradation"].idxmax()
    return top_value, merged


def find_root_cause(current_df: pd.DataFrame, baseline_df: pd.DataFrame) -> dict:
    """
    Run the worst-offender analysis across every candidate dimension and
    also surface the dominant failure_reason among the failed transactions.
    """
    root_cause = {}
    concentration_detail = {}

    for dim in SEGMENT_DIMENSIONS:
        top_value, breakdown = worst_offender(current_df, baseline_df, dim)
        root_cause[f"top_{dim}"] = top_value
        if breakdown is not None:
            concentration_detail[dim] = (
                breakdown.sort_values("degradation", ascending=False)
                .round(2)
                .to_dict(orient="index")
            )

    # Failure reason: just the most common reason among the failures in
    # the current window — directly tells us the mechanism (TIMEOUT, etc.)
    failed_in_window = current_df.loc[~current_df["is_success"], "failure_reason"]
    top_failure_reason = failed_in_window.mode().iloc[0] if not failed_in_window.empty else None
    root_cause["top_failure_reason"] = top_failure_reason

    root_cause["_concentration_detail"] = concentration_detail

    # -----------------------------------------------------------------
    # Joint segment drill-down: a fire is often localized to a SPECIFIC
    # combination (e.g. "HDFC + UPI"), which can look diluted/small at
    # the whole-portfolio level even though that segment itself is
    # cratering. Combine the two strongest single-dimension suspects
    # (payment_method + bank) and measure that exact segment directly.
    # -----------------------------------------------------------------
    combo_stats = find_worst_combo(
        current_df, baseline_df, dims=("payment_method", "bank"), min_count=MIN_SEGMENT_TXNS
    )
    root_cause["segment_analysis"] = combo_stats

    return root_cause


def find_worst_combo(current_df: pd.DataFrame, baseline_df: pd.DataFrame,
                      dims=("payment_method", "bank"), min_count: int = MIN_SEGMENT_TXNS) -> dict:
    """
    Find the single (dim1, dim2) combination — e.g. a specific
    payment_method + bank pair — whose success rate degraded the most
    vs. its own history, and report its stats directly. This is what
    lets us say "HDFC + UPI specifically dropped from 94% to 39%" even
    when the portfolio-wide number barely moves.
    """
    dims = list(dims)

    cur = current_df.groupby(dims).agg(total=("transaction_id", "count"), success=("is_success", "sum"))
    cur = cur[cur["total"] >= min_count]
    if cur.empty:
        return {}
    cur["current_rate"] = 100 * cur["success"] / cur["total"]

    base = baseline_df.groupby(dims).agg(total=("transaction_id", "count"), success=("is_success", "sum"))
    base["baseline_rate"] = 100 * base["success"] / base["total"]

    merged = cur.join(base[["baseline_rate"]], how="left")
    merged["baseline_rate"] = merged["baseline_rate"].fillna(merged["current_rate"])
    merged["degradation"] = merged["baseline_rate"] - merged["current_rate"]

    if merged.empty:
        return {}

    worst_key = merged["degradation"].idxmax()
    row = merged.loc[worst_key]
    # idxmax on a MultiIndex-grouped frame returns numpy scalars/tuples;
    # normalize to plain Python types so the result stays JSON-friendly.
    if isinstance(worst_key, tuple):
        worst_key = tuple(k.item() if hasattr(k, "item") else k for k in worst_key)
    elif hasattr(worst_key, "item"):
        worst_key = worst_key.item()

    # Failed GMV for exactly this segment in the current window
    if len(dims) == 1:
        seg_mask = current_df[dims[0]] == worst_key
        segment_label = {dims[0]: worst_key}
    else:
        seg_mask = np.logical_and.reduce([current_df[d] == v for d, v in zip(dims, worst_key)])
        segment_label = dict(zip(dims, worst_key))

    seg_df = current_df.loc[seg_mask]
    seg_failed_gmv = float(seg_df.loc[~seg_df["is_success"], "amount"].sum())
    seg_drop_pct = round(
        100 * (row["baseline_rate"] - row["current_rate"]) / row["baseline_rate"], 2
    ) if row["baseline_rate"] else 0.0

    return {
        "segment": segment_label,
        "segment_total_txns": int(row["total"]),
        "segment_failed_txns": int(row["total"] - row["success"]),
        "segment_baseline_success_rate": round(float(row["baseline_rate"]), 2),
        "segment_current_success_rate": round(float(row["current_rate"]), 2),
        "segment_drop_percentage": seg_drop_pct,
        "segment_revenue_at_risk": round(seg_failed_gmv, 2),
    }


# ---------------------------------------------------------------------------
# 5. Simple demo / run script
# ---------------------------------------------------------------------------
def print_incident_report(incident: dict) -> None:
    print("=" * 60)
    print("FINANCIAL FIREFIGHTER — STEP 2: FIRE DETECTOR")
    print("=" * 60)

    if not incident.get("incident_detected"):
        print("\n✅ No incident detected.")
        if "reason" in incident:
            print(f"   ({incident['reason']})")
        else:
            print(f"   Current window success rate: {incident.get('current_success_rate')}% "
                  f"(baseline: {incident.get('baseline_success_rate')}%)")
        return

    print(f"\n🔥 INCIDENT DETECTED — severity: {incident['severity']}")
    print(f"   Window analyzed       : {incident['current_window_start']} "
          f"(+{incident['window_size']})")
    print(f"   Baseline success rate : {incident['baseline_success_rate']}%")
    print(f"   Current success rate  : {incident['current_success_rate']}%")
    print(f"   Relative drop         : {incident['drop_percentage']}%")
    print(f"   Z-score               : {incident['z_score']}")
    print(f"   Affected transactions : {incident['affected_transactions']:,}")
    print(f"   Revenue at risk       : ₹{incident['revenue_at_risk']:,.2f}")

    print("\n   --- Suspected root cause (worst-degraded segment per dimension) ---")
    print(f"   Payment method : {incident['top_payment_method']}")
    print(f"   Bank           : {incident['top_bank']}")
    print(f"   Device         : {incident['top_device']}")
    print(f"   City           : {incident['top_city']}")
    print(f"   Failure reason : {incident['top_failure_reason']}")

    seg = incident.get("segment_analysis")
    if seg:
        print("\n   --- Joint segment drill-down (the exact 'seat of the fire') ---")
        print(f"   Segment                : {seg['segment']}")
        print(f"   Baseline success rate  : {seg['segment_baseline_success_rate']}%")
        print(f"   Current success rate   : {seg['segment_current_success_rate']}%")
        print(f"   Segment drop           : {seg['segment_drop_percentage']}%")
        print(f"   Segment txns (current) : {seg['segment_total_txns']} "
              f"({seg['segment_failed_txns']} failed)")
        print(f"   Segment revenue at risk: ₹{seg['segment_revenue_at_risk']:,.2f}")
        if "portfolio_severity" in incident:
            print(f"\n   ⚠️  Severity escalated from portfolio-level "
                  f"'{incident['portfolio_severity']}' → '{incident['severity']}' "
                  f"because this segment is degrading far worse than the overall numbers show.")


if __name__ == "__main__":
    data = load_data(CSV_PATH)
    incident = detect_incident(data)
    print_incident_report(incident)

    print("\n--- Raw incident object ---")
    # Hide the verbose per-segment detail dict in the printed summary
    clean = {k: v for k, v in incident.items() if k != "_concentration_detail"}
    for k, v in clean.items():
        print(f"  {k!r}: {v!r}")
