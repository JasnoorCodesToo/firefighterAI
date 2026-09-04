"""
Financial Firefighter — STEP 3: AI INVESTIGATOR
====================================================
Takes the structured incident object produced by detector.py and turns
it into an investigation report: root cause, supporting evidence,
a bounded recommended action, a confidence score, and whether a human
needs to sign off before anything gets executed.

DESIGN PRINCIPLE — numbers vs. narrative are strictly separated:
  - Every NUMBER in the output (revenue_at_risk, confidence_score,
    affected_segments, comparisons across banks/methods/devices/etc.)
    is computed directly in Python from the raw dataframe + the
    detector's output. The LLM never sees a blank canvas to invent
    figures on.
  - The LLM (if an API key is available) is used ONLY to turn that
    already-computed evidence into a readable summary / root-cause
    explanation, under a prompt that explicitly forbids introducing
    new statistics. If no API key is available, a deterministic
    template-based fallback produces equivalent (if less fluent)
    prose, so this module always runs end-to-end without external
    dependencies.

Run:
    python3 investigator.py
"""

import os
import json
import numpy as np
import pandas as pd
import requests

from analyze import load_data, CSV_PATH
from detector import detect_incident, SEGMENT_DIMENSIONS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-4-6"
LLM_MAX_TOKENS = 700

# Confidence below this -> we don't trust the diagnosis enough to act alone
CONFIDENCE_HUMAN_APPROVAL_THRESHOLD = 75

# How many extra minutes of segment-level history to scan when estimating
# when the incident actually started (onset detection)
ONSET_LOOKBACK_WINDOWS = 8
ONSET_WINDOW_FREQ = "5min"

FAILURE_REASON_PLAYBOOK = {
    "TIMEOUT": (
        "a latency / connectivity problem on the processing rail for this segment "
        "(requests are being accepted but not completing in time)"
    ),
    "BANK_DECLINE": (
        "an elevated authorization-decline rate from the issuing/acquiring bank side "
        "of this segment"
    ),
    "TECHNICAL_ERROR": (
        "a technical fault in the integration for this segment, consistent with a "
        "recent deployment, config change, or upstream outage"
    ),
    "INSUFFICIENT_FUNDS": (
        "customer-side failures (insufficient funds), which is normal transaction "
        "friction rather than a system incident"
    ),
}

RECOMMENDED_ACTION_PLAYBOOK = {
    "TIMEOUT": (
        "Fail over new traffic for {segment_desc} to an alternate processor/bank route "
        "for the next {window} while the rail recovers, and open a P1 ticket with the "
        "relevant bank's technical team."
    ),
    "BANK_DECLINE": (
        "Temporarily reduce routing weight to {segment_desc} and shift volume to "
        "alternate banks for this payment method; notify the bank's relationship "
        "manager to confirm whether they are aware of an issue."
    ),
    "TECHNICAL_ERROR": (
        "Pause new transaction attempts on {segment_desc} pending a technical health "
        "check; check for recent deployments/config changes on this integration before "
        "re-enabling."
    ),
    "INSUFFICIENT_FUNDS": (
        "No system-level action required — this pattern looks like normal customer-side "
        "decline behavior. Continue monitoring only."
    ),
    "_default": (
        "Escalate {segment_desc} to the on-call payments engineer for manual "
        "investigation; do not take automated action until root cause is confirmed."
    ),
}


# ---------------------------------------------------------------------------
# 1. Evidence gathering — everything here is plain Python/pandas arithmetic,
#    no LLM involved. This is the ONLY source of numbers in the final report.
# ---------------------------------------------------------------------------
def _resolve_affected_segment(incident: dict) -> dict:
    """
    Figure out which segment (dict of {dimension: value}) is under
    investigation. Prefer the detector's joint segment_analysis (most
    specific — e.g. {'payment_method': 'UPI', 'bank': 'HDFC'}); fall back
    to whichever single dimensions the detector flagged as top offenders.
    """
    seg_analysis = incident.get("segment_analysis") or {}
    if seg_analysis.get("segment"):
        return dict(seg_analysis["segment"])

    # Fallback: use whichever single-dimension "top_*" fields exist
    segment = {}
    for dim in SEGMENT_DIMENSIONS:
        val = incident.get(f"top_{dim}")
        if val is not None:
            segment[dim] = val
    return segment


def _segment_mask(df: pd.DataFrame, segment: dict) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for dim, val in segment.items():
        mask &= df[dim] == val
    return mask


def compare_against_siblings(df: pd.DataFrame, incident: dict, segment: dict) -> dict:
    """
    For each dimension in the affected segment, compare its current-window
    success rate against the OTHER values of that same dimension (e.g.
    HDFC vs. every other bank) in the same window. This is what lets us
    say "this is specific to HDFC, not a UPI-wide problem" (or vice versa).
    """
    window_start = pd.Timestamp(incident["current_window_start"])
    window_end = pd.Timestamp(incident["current_window_end"])
    current_df = df[(df["timestamp"] > window_start) & (df["timestamp"] <= window_end)]

    comparisons = {}
    for dim, affected_value in segment.items():
        grp = current_df.groupby(dim).agg(total=("transaction_id", "count"), success=("is_success", "sum"))
        grp["success_rate"] = round(100 * grp["success"] / grp["total"], 2)

        affected_rate = float(grp.loc[affected_value, "success_rate"]) if affected_value in grp.index else None
        others = grp.drop(index=affected_value, errors="ignore")
        others_avg_rate = round(float(others["success_rate"].mean()), 2) if not others.empty else None

        comparisons[dim] = {
            "affected_value": affected_value,
            "affected_success_rate": affected_rate,
            "other_values_avg_success_rate": others_avg_rate,
            "other_values_breakdown": others["success_rate"].round(2).to_dict(),
        }

    return comparisons


def analyze_device_pattern(df: pd.DataFrame, incident: dict, segment: dict) -> dict:
    """Is the incident concentrated on a specific device within the segment, or spread evenly?"""
    window_start = pd.Timestamp(incident["current_window_start"])
    window_end = pd.Timestamp(incident["current_window_end"])
    seg_df = df[_segment_mask(df, segment) & (df["timestamp"] > window_start) & (df["timestamp"] <= window_end)]

    if seg_df.empty:
        return {}

    grp = seg_df.groupby("device").agg(total=("transaction_id", "count"), success=("is_success", "sum"))
    grp["success_rate"] = round(100 * grp["success"] / grp["total"], 2)
    return grp[["total", "success_rate"]].to_dict(orient="index")


def analyze_failure_reasons(df: pd.DataFrame, incident: dict, segment: dict) -> dict:
    """What is actually going wrong for the failed transactions in this segment?"""
    window_start = pd.Timestamp(incident["current_window_start"])
    window_end = pd.Timestamp(incident["current_window_end"])
    seg_df = df[_segment_mask(df, segment) & (df["timestamp"] > window_start) & (df["timestamp"] <= window_end)]
    failed = seg_df.loc[~seg_df["is_success"], "failure_reason"]

    if failed.empty:
        return {}

    counts = failed.value_counts()
    pct = round(100 * counts / counts.sum(), 1)
    return {
        "counts": counts.to_dict(),
        "pct": pct.to_dict(),
        "dominant_reason": counts.idxmax(),
        "dominant_reason_share_pct": float(pct.max()),
    }


def analyze_timing(df: pd.DataFrame, incident: dict, segment: dict,
                    freq: str = ONSET_WINDOW_FREQ, lookback_windows: int = ONSET_LOOKBACK_WINDOWS) -> dict:
    """
    Trace the affected segment's own success rate over recent time (finer
    granularity than the detector's main window) to describe how the
    incident evolved: sudden cliff vs. gradual decay, and roughly when it
    started.
    """
    window_end = pd.Timestamp(incident["current_window_end"])
    seg_df = df[_segment_mask(df, segment)]
    if seg_df.empty:
        return {}

    recent = seg_df[seg_df["timestamp"] <= window_end].set_index("timestamp")
    windows = recent["is_success"].resample(freq).agg(["mean", "count"])
    windows = windows[windows["count"] > 0].tail(lookback_windows)
    windows["success_rate"] = round(100 * windows["mean"], 2)

    trend = [
        {"window_start": str(idx), "success_rate": float(row["success_rate"]), "txns": int(row["count"])}
        for idx, row in windows.iterrows()
    ]

    # Rough onset estimate: walk backwards from the most recent window and
    # find the earliest point in an unbroken run of "degraded" windows
    # (below the segment's own baseline).
    baseline_rate = None
    seg_analysis = incident.get("segment_analysis") or {}
    if seg_analysis.get("segment") == segment:
        baseline_rate = seg_analysis.get("segment_baseline_success_rate")

    onset_window = None
    if baseline_rate:
        degraded_cutoff = baseline_rate * 0.85  # meaningfully below normal
        for point in reversed(trend):
            if point["success_rate"] <= degraded_cutoff:
                onset_window = point["window_start"]
            else:
                break

    return {"recent_trend": trend, "onset_window_estimate": onset_window}


def gather_evidence(df: pd.DataFrame, incident: dict) -> dict:
    """Top-level evidence-gathering pass — everything computed here is a plain fact."""
    segment = _resolve_affected_segment(incident)

    return {
        "segment": segment,
        "sibling_comparison": compare_against_siblings(df, incident, segment),
        "device_pattern": analyze_device_pattern(df, incident, segment),
        "failure_reasons": analyze_failure_reasons(df, incident, segment),
        "timing": analyze_timing(df, incident, segment),
    }


# ---------------------------------------------------------------------------
# 2. Confidence scoring — rule-based, deterministic, no LLM.
# ---------------------------------------------------------------------------
def compute_confidence_score(incident: dict, evidence: dict) -> int:
    """
    Blend several independent numerical signals into a 0-100 confidence
    that we've correctly identified the root cause / segment:

      - statistical significance (z-score of the portfolio-level anomaly)
      - magnitude of the segment's own drop vs. its baseline
      - sample size backing the segment (small samples = less trust)
      - how isolated the problem is (affected segment vs. sibling average
        gap — a big gap means we've correctly localized it, not just
        caught general noise)
      - concentration of a single failure_reason (one dominant cause is
        more diagnosable than a scattered mix)
    """
    score = 0.0

    z = abs(incident.get("z_score", 0) or 0)
    score += min(z / 5.0, 1.0) * 25  # up to 25 pts, saturates around z=5

    seg = incident.get("segment_analysis") or {}
    seg_drop = seg.get("segment_drop_percentage", 0) or 0
    score += min(seg_drop / 60.0, 1.0) * 25  # up to 25 pts, saturates around a 60% drop

    seg_n = seg.get("segment_total_txns", 0) or 0
    score += min(seg_n / 50.0, 1.0) * 20  # up to 20 pts, saturates at 50+ txns in segment

    sibling = evidence.get("sibling_comparison", {})
    isolation_gaps = []
    for dim_info in sibling.values():
        affected = dim_info.get("affected_success_rate")
        others = dim_info.get("other_values_avg_success_rate")
        if affected is not None and others:
            isolation_gaps.append(max(others - affected, 0))
    if isolation_gaps:
        avg_gap = sum(isolation_gaps) / len(isolation_gaps)
        score += min(avg_gap / 40.0, 1.0) * 20  # up to 20 pts, saturates around a 40pt gap

    fr = evidence.get("failure_reasons", {})
    dominant_share = fr.get("dominant_reason_share_pct", 0) or 0
    score += min(dominant_share / 100.0, 1.0) * 10  # up to 10 pts

    return int(round(min(score, 100)))


# ---------------------------------------------------------------------------
# 3. Recommended action — deterministic template selection based on the
#    dominant failure reason. The wording is generic; only the *inputs*
#    (segment description, window size, failure reason) come from data.
# ---------------------------------------------------------------------------
def describe_segment(segment: dict) -> str:
    if not segment:
        return "the affected segment"
    return " + ".join(f"{v} ({k})" for k, v in segment.items())


def recommend_action(incident: dict, evidence: dict, segment: dict) -> str:
    fr = evidence.get("failure_reasons", {})
    dominant_reason = fr.get("dominant_reason") or incident.get("top_failure_reason")
    template = RECOMMENDED_ACTION_PLAYBOOK.get(dominant_reason, RECOMMENDED_ACTION_PLAYBOOK["_default"])
    return template.format(segment_desc=describe_segment(segment), window=incident.get("window_size", "the incident window"))


# ---------------------------------------------------------------------------
# 4. LLM layer — narrative only, strictly grounded in the evidence dict.
# ---------------------------------------------------------------------------
def build_llm_prompt(incident: dict, evidence: dict, confidence_score: int, recommended_action: str) -> str:
    facts = {
        "segment": evidence["segment"],
        "severity": incident.get("severity"),
        "baseline_success_rate_portfolio": incident.get("baseline_success_rate"),
        "current_success_rate_portfolio": incident.get("current_success_rate"),
        "segment_analysis": incident.get("segment_analysis"),
        "sibling_comparison": evidence.get("sibling_comparison"),
        "device_pattern": evidence.get("device_pattern"),
        "failure_reasons": evidence.get("failure_reasons"),
        "timing": evidence.get("timing"),
        "confidence_score": confidence_score,
        "recommended_action": recommended_action,
    }

    return f"""You are a payments incident analyst. Below is a JSON object of PRE-COMPUTED,
VERIFIED facts about a live payment success-rate incident. Do not invent, estimate,
or alter any number — only use numbers exactly as given below.

FACTS:
{json.dumps(facts, indent=2, default=str)}

Write a JSON object with exactly two keys:
  "incident_summary": a 2-3 sentence plain-English summary of what is happening,
      referencing the affected segment and the scale of the drop using ONLY the
      numbers given above.
  "probable_root_cause": 2-3 sentences explaining the most likely root cause,
      grounded in the failure_reasons, device_pattern, sibling_comparison, and
      timing evidence above. Be specific about WHY the evidence points there,
      but do not cite any number not present in the facts above.

Respond with ONLY the JSON object, no other text."""


def call_llm(prompt: str) -> dict | None:
    """
    Call the Anthropic API for the narrative fields. Returns None (triggering
    the deterministic fallback) if no API key is configured or the call fails
    for any reason — this keeps the module runnable in any environment.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        resp = requests.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": LLM_MAX_TOKENS,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        text = "".join(
            block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
        ).strip()
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(text)
    except Exception as e:  # noqa: BLE001 - any failure here should just trigger fallback
        print(f"[investigator] LLM call unavailable/failed ({e}); using template fallback.")
        return None


def fallback_narrative(incident: dict, evidence: dict, confidence_score: int) -> dict:
    """Deterministic, template-based stand-in for the LLM narrative fields."""
    segment = evidence["segment"]
    seg_desc = describe_segment(segment)
    seg = incident.get("segment_analysis") or {}
    fr = evidence.get("failure_reasons", {})
    dominant_reason = fr.get("dominant_reason") or incident.get("top_failure_reason") or "an unidentified cause"

    if seg:
        summary = (
            f"Payment success rate for {seg_desc} dropped to "
            f"{seg.get('segment_current_success_rate', incident.get('current_success_rate'))}% "
            f"in the current window, down from a baseline of "
            f"{seg.get('segment_baseline_success_rate', incident.get('baseline_success_rate'))}% "
            f"— a {seg.get('segment_drop_percentage', incident.get('drop_percentage'))}% relative drop, "
            f"affecting {seg.get('segment_failed_txns', incident.get('affected_transactions'))} transactions "
            f"worth an estimated ₹{seg.get('segment_revenue_at_risk', incident.get('revenue_at_risk')):,.0f} at risk."
        )
    else:
        summary = (
            f"Overall payment success rate dropped to {incident.get('current_success_rate')}% "
            f"from a baseline of {incident.get('baseline_success_rate')}%."
        )

    cause_explanation = FAILURE_REASON_PLAYBOOK.get(
        dominant_reason, "an as-yet-unclassified failure pattern that warrants manual review"
    )
    dominant_share = fr.get("dominant_reason_share_pct")
    share_clause = f" ({dominant_share}% of failures in this segment)" if dominant_share else ""

    root_cause = (
        f"The dominant failure reason is '{dominant_reason}'{share_clause}, consistent with {cause_explanation}. "
        f"This pattern is concentrated in {seg_desc} rather than spread evenly across other segments, "
        f"pointing to a localized issue specific to this route rather than a platform-wide problem."
    )

    return {"incident_summary": summary, "probable_root_cause": root_cause}


# ---------------------------------------------------------------------------
# 5. Human approval gate
# ---------------------------------------------------------------------------
def requires_human_approval(confidence_score: int, severity: str) -> bool:
    """
    Conservative gate: low confidence always needs a human. High-severity
    incidents also default to requiring approval before any automated
    action touches real money/routing, even if confidence is high — the
    bounded action is only ever a *recommendation* until Step 4 executes
    it under a human-approved policy.
    """
    if confidence_score < CONFIDENCE_HUMAN_APPROVAL_THRESHOLD:
        return True
    if severity in ("CRITICAL", "HIGH"):
        return True
    return False


# ---------------------------------------------------------------------------
# 6. Top-level orchestration
# ---------------------------------------------------------------------------
def investigate(incident: dict, df: pd.DataFrame) -> dict:
    """
    Main entry point: given a (possibly no-incident) detector output and
    the raw dataframe, produce the structured investigation report.
    """
    if not incident.get("incident_detected"):
        return {
            "incident_summary": "No active incident detected — payment success rates are within normal range.",
            "probable_root_cause": None,
            "supporting_evidence": [],
            "affected_segments": [],
            "revenue_at_risk": 0,
            "recommended_action": "No action needed. Continue routine monitoring.",
            "confidence_score": 100,
            "severity": "NONE",
            "human_approval_required": False,
        }

    evidence = gather_evidence(df, incident)
    segment = evidence["segment"]

    confidence_score = compute_confidence_score(incident, evidence)
    action = recommend_action(incident, evidence, segment)
    approval_required = requires_human_approval(confidence_score, incident.get("severity", "NONE"))

    # Numeric revenue-at-risk: prefer the precise segment-level figure,
    # fall back to the detector's whole-window figure if unavailable.
    seg_analysis = incident.get("segment_analysis") or {}
    revenue_at_risk = seg_analysis.get("segment_revenue_at_risk", incident.get("revenue_at_risk"))

    # supporting_evidence: plain Python-built facts, NOT LLM-generated,
    # so every figure quoted here is guaranteed to trace back to the data.
    supporting_evidence = build_supporting_evidence(incident, evidence)

    # Narrative fields: LLM if available, else deterministic fallback.
    prompt = build_llm_prompt(incident, evidence, confidence_score, action)
    llm_result = call_llm(prompt)
    if llm_result and "incident_summary" in llm_result and "probable_root_cause" in llm_result:
        narrative = llm_result
    else:
        narrative = fallback_narrative(incident, evidence, confidence_score)

    return {
        "incident_summary": narrative["incident_summary"],
        "probable_root_cause": narrative["probable_root_cause"],
        "supporting_evidence": supporting_evidence,
        "affected_segments": list(segment.values()),
        "revenue_at_risk": round(float(revenue_at_risk), 2) if revenue_at_risk is not None else None,
        "recommended_action": action,
        "confidence_score": confidence_score,
        "severity": incident.get("severity"),
        "human_approval_required": approval_required,
        # Extra detail kept for downstream steps / debugging, prefixed with
        # "_" so UI layers know it's supplementary, not part of the core schema.
        "_evidence": evidence,
    }


def build_supporting_evidence(incident: dict, evidence: dict) -> list:
    """Turn the computed evidence dict into a list of plain-fact strings."""
    facts = []
    seg = incident.get("segment_analysis") or {}
    if seg:
        facts.append(
            f"{describe_segment(evidence['segment'])} success rate fell from "
            f"{seg.get('segment_baseline_success_rate')}% (baseline) to "
            f"{seg.get('segment_current_success_rate')}% in the current window "
            f"({seg.get('segment_drop_percentage')}% relative drop)."
        )

    for dim, info in evidence.get("sibling_comparison", {}).items():
        if info.get("affected_success_rate") is not None and info.get("other_values_avg_success_rate") is not None:
            facts.append(
                f"{info['affected_value']} ({dim}) is at {info['affected_success_rate']}% success, "
                f"vs. {info['other_values_avg_success_rate']}% average across other {dim} values in the same window."
            )

    fr = evidence.get("failure_reasons", {})
    if fr.get("dominant_reason"):
        facts.append(
            f"Dominant failure reason within the segment is '{fr['dominant_reason']}', "
            f"accounting for {fr['dominant_reason_share_pct']}% of its failures."
        )

    device_pattern = evidence.get("device_pattern", {})
    if device_pattern:
        worst_device = min(device_pattern.items(), key=lambda kv: kv[1]["success_rate"])
        facts.append(
            f"Within the segment, '{worst_device[0]}' devices show the lowest success rate "
            f"({worst_device[1]['success_rate']}%, n={worst_device[1]['total']})."
        )

    timing = evidence.get("timing", {})
    if timing.get("onset_window_estimate"):
        facts.append(f"Degradation appears to have started around {timing['onset_window_estimate']}.")

    facts.append(
        f"Portfolio-wide, success rate is {incident.get('current_success_rate')}% vs. a "
        f"{incident.get('baseline_success_rate')}% baseline (z-score {incident.get('z_score')})."
    )

    return facts


# ---------------------------------------------------------------------------
# 7. Demo / run script
# ---------------------------------------------------------------------------
def print_investigation_report(report: dict) -> None:
    print("=" * 60)
    print("FINANCIAL FIREFIGHTER — STEP 3: AI INVESTIGATOR")
    print("=" * 60)

    print(f"\nSeverity              : {report['severity']}")
    print(f"Confidence score       : {report['confidence_score']}/100")
    print(f"Human approval required: {report['human_approval_required']}")
    print(f"Affected segments      : {report['affected_segments']}")
    rev = report["revenue_at_risk"]
    print(f"Revenue at risk        : ₹{rev:,.2f}" if rev else "Revenue at risk        : ₹0")

    print(f"\nIncident summary:\n  {report['incident_summary']}")
    if report["probable_root_cause"]:
        print(f"\nProbable root cause:\n  {report['probable_root_cause']}")

    if report["supporting_evidence"]:
        print("\nSupporting evidence:")
        for fact in report["supporting_evidence"]:
            print(f"  • {fact}")

    print(f"\nRecommended action:\n  {report['recommended_action']}")


if __name__ == "__main__":
    data = load_data(CSV_PATH)
    incident = detect_incident(data)
    report = investigate(incident, data)
    print_investigation_report(report)

    print("\n--- Investigation report (JSON, core schema only) ---")
    core_schema = {k: v for k, v in report.items() if not k.startswith("_")}
    print(json.dumps(core_schema, indent=2, default=str))
