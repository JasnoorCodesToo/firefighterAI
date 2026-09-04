"""
Financial Firefighter — STEP 4: THE FIREFIGHTER AGENT
==========================================================
Takes the investigator's diagnosis (+ the original detector incident, for
window/segment numeric context, + the raw dataframe, for simulation) and
decides whether — and how — to respond.

This module NEVER touches a real payment system. Everything here is a
SIMULATION: we construct a synthetic "post-intervention" copy of the
affected transactions to model what would happen if traffic were
rerouted/reduced/rolled back, and measure the effect on that copy.

DESIGN PRINCIPLE — same separation as investigator.py:
  - Every number (pre/post success rate, revenue protected, transactions
    recovered) is computed from real transaction rows in the dataframe —
    either ACTUAL rows (pre-intervention, revenue at risk) or a
    deterministically-constructed SIMULATED copy (post-intervention).
  - Nothing here calls an LLM. This module is pure decision-logic +
    arithmetic, which is exactly what should gate anything touching money.

GUARDRAILS (hard-coded safety rules, not data-derived — these are policy):
  - confidence_score < 75          -> never auto-execute
  - severity in (HIGH, CRITICAL)   -> never auto-execute
  - max intervention duration      -> 30 minutes
  - no improvement after checkpoint-> roll back
  - success rate near baseline     -> stop / mark resolved

Run:
    python3 firefighter.py
"""

import json
from datetime import datetime, timezone

import pandas as pd

from analyze import load_data, CSV_PATH
from detector import detect_incident, SEGMENT_DIMENSIONS
from investigator import investigate, describe_segment

# ---------------------------------------------------------------------------
# Config — guardrails & simulation assumptions (explicit, documented, tunable)
# ---------------------------------------------------------------------------
CONFIDENCE_AUTO_EXECUTE_MIN = 75          # guardrail: min confidence to auto-act
BLOCKED_SEVERITIES = {"HIGH", "CRITICAL"}  # guardrail: never auto-act on these

MAX_INTERVENTION_MINUTES = 30             # guardrail: hard cap on intervention duration
CHECK_INTERVAL_MINUTES = 5                # how often we "check in" during the simulation
RAMP_MINUTES = 15                         # modeling assumption: time for full effect to land
RESOLVED_RATIO = 0.95                     # "close to baseline" = within 95% of it

ALLOWED_INTERVENTIONS = {
    "REDIRECT_TRAFFIC",
    "REDUCE_AFFECTED_ROUTE",
    "MONITOR",
    "ESCALATE_TO_HUMAN",
    "ROLLBACK",
}

# Deterministic mapping: dominant failure_reason -> intervention type.
# This is a policy table, not a fact derived from this specific dataset,
# so it never references any specific bank/method name.
INTERVENTION_FOR_REASON = {
    "TIMEOUT": "REDIRECT_TRAFFIC",
    "BANK_DECLINE": "REDUCE_AFFECTED_ROUTE",
    "TECHNICAL_ERROR": "ROLLBACK",
    "INSUFFICIENT_FUNDS": "MONITOR",
}

# Modeling assumption: how much of the gap to the "healthy" (sibling-average)
# success rate each intervention type is assumed to close. This is a
# documented simulation parameter, not a measured fact — it drives the
# SIMULATED post-intervention numbers, which are clearly labeled as such.
INTERVENTION_EFFECTIVENESS = {
    "REDIRECT_TRAFFIC": 0.90,
    "REDUCE_AFFECTED_ROUTE": 0.60,
    "ROLLBACK": 0.85,
    "MONITOR": 0.0,
    "ESCALATE_TO_HUMAN": 0.0,
}


# ---------------------------------------------------------------------------
# Small helper: audit log
# ---------------------------------------------------------------------------
def _log(audit_log: list, action: str, reason: str, **extra) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "action": action,
        "reason": reason,
    }
    entry.update(extra)
    audit_log.append(entry)


# ---------------------------------------------------------------------------
# 1. Guardrail evaluation
# ---------------------------------------------------------------------------
def evaluate_guardrails(investigation: dict) -> dict:
    """
    Independently re-check the safety gates (defense in depth — even if
    investigator.py already flagged human_approval_required, the
    firefighter re-verifies before touching anything).
    """
    reasons = []
    confidence = investigation.get("confidence_score", 0)
    severity = investigation.get("severity", "NONE")

    if confidence < CONFIDENCE_AUTO_EXECUTE_MIN:
        reasons.append(f"confidence_score {confidence} is below the auto-execute minimum ({CONFIDENCE_AUTO_EXECUTE_MIN}).")
    if severity in BLOCKED_SEVERITIES:
        reasons.append(f"severity '{severity}' is in the blocked-for-auto-execution list ({sorted(BLOCKED_SEVERITIES)}).")
    if investigation.get("human_approval_required"):
        reasons.append("investigator flagged human_approval_required=True.")

    return {"can_auto_execute": len(reasons) == 0, "blocking_reasons": reasons}


# ---------------------------------------------------------------------------
# 2. Intervention selection
# ---------------------------------------------------------------------------
def select_intervention(investigation: dict) -> str:
    """
    Pick an intervention type from the ALLOWED set, deterministically,
    based on the investigator's dominant failure_reason evidence (falls
    back to ESCALATE_TO_HUMAN if the evidence isn't conclusive enough to
    pick a specific fix).
    """
    evidence = investigation.get("_evidence", {})
    dominant_reason = (evidence.get("failure_reasons") or {}).get("dominant_reason")

    intervention = INTERVENTION_FOR_REASON.get(dominant_reason, "ESCALATE_TO_HUMAN")
    assert intervention in ALLOWED_INTERVENTIONS
    return intervention


def pick_recovery_dimension(evidence: dict) -> tuple:
    """
    Of the dimensions in the affected segment (e.g. payment_method, bank),
    find the one where the affected value is furthest below its siblings'
    average — that's the dimension whose "healthy" average we use as the
    simulated recovery target (i.e. "what if this route behaved like a
    typical alternate route").
    Returns (dimension, affected_value, sibling_avg_rate) or (None, None, None).
    """
    sibling = evidence.get("sibling_comparison", {})
    best = None
    for dim, info in sibling.items():
        affected = info.get("affected_success_rate")
        others = info.get("other_values_avg_success_rate")
        if affected is None or others is None:
            continue
        gap = others - affected
        if best is None or gap > best[3]:
            best = (dim, info["affected_value"], others, gap)

    if best is None:
        return None, None, None
    return best[0], best[1], best[2]


# ---------------------------------------------------------------------------
# 3. Simulated intervention — builds an actual synthetic post-intervention
#    dataset by deterministically "recovering" a subset of failed
#    transactions, rather than just faking an aggregate number.
# ---------------------------------------------------------------------------
def get_segment_window_df(df: pd.DataFrame, incident: dict, segment: dict) -> pd.DataFrame:
    window_start = pd.Timestamp(incident["current_window_start"])
    window_end = pd.Timestamp(incident["current_window_end"])
    mask = (df["timestamp"] > window_start) & (df["timestamp"] <= window_end)
    for dim, val in segment.items():
        mask &= df[dim] == val
    return df.loc[mask].copy()


def simulate_intervention(segment_df: pd.DataFrame, target_rate: float) -> dict:
    """
    Deterministically construct a simulated "after intervention" copy of
    the segment's current-window transactions: the earliest-failing
    transactions are the first to be "fixed" as the intervention takes
    effect (a reasonable, reproducible ordering — not random).

    Returns the simulated dataframe plus the concrete numbers derived
    from it (achieved rate, recovered transactions, recovered revenue —
    the exact sum of the real `amount` values of the flipped rows).
    """
    total = len(segment_df)
    if total == 0:
        return {
            "simulated_df": segment_df,
            "achieved_rate": 0.0,
            "recovered_ids": [],
            "recovered_amount": 0.0,
        }

    current_success = int(segment_df["is_success"].sum())
    target_success = round(target_rate / 100 * total)
    to_flip = max(target_success - current_success, 0)

    failed_sorted = segment_df.loc[~segment_df["is_success"]].sort_values("timestamp")
    flip_ids = failed_sorted["transaction_id"].head(to_flip).tolist()

    sim_df = segment_df.copy()
    sim_df["simulated"] = False
    flip_mask = sim_df["transaction_id"].isin(flip_ids)
    sim_df.loc[flip_mask, "is_success"] = True
    sim_df.loc[flip_mask, "simulated"] = True

    achieved_success = current_success + len(flip_ids)
    achieved_rate = round(100 * achieved_success / total, 2)
    recovered_amount = round(float(segment_df.loc[segment_df["transaction_id"].isin(flip_ids), "amount"].sum()), 2)

    return {
        "simulated_df": sim_df,
        "achieved_rate": achieved_rate,
        "recovered_ids": flip_ids,
        "recovered_amount": recovered_amount,
    }


# ---------------------------------------------------------------------------
# 4. Checkpoint / ramp simulation — models the intervention "landing" over
#    time, evaluated every CHECK_INTERVAL_MINUTES up to the max duration.
# ---------------------------------------------------------------------------
def run_checkpoints(pre_rate: float, post_rate: float, baseline_rate: float, audit_log: list) -> dict:
    """
    Step through simulated time in CHECK_INTERVAL_MINUTES increments (up to
    MAX_INTERVENTION_MINUTES), modeling the success rate ramping linearly
    from pre_rate to post_rate over RAMP_MINUTES. Stops early ("resolved")
    once the rate is back within RESOLVED_RATIO of baseline; rolls back
    immediately if there's no improvement at all.
    """
    if post_rate <= pre_rate:
        _log(audit_log, "ROLLBACK_TRIGGERED",
             f"No improvement observed (post-intervention model rate {post_rate}% <= "
             f"pre-intervention {pre_rate}%). Rolling back per guardrail policy.")
        return {"status": "ROLLED_BACK", "resolution_time_minutes": CHECK_INTERVAL_MINUTES,
                "final_rate": pre_rate}

    threshold = round(baseline_rate * RESOLVED_RATIO, 2)
    t = 0
    while t < MAX_INTERVENTION_MINUTES:
        t += CHECK_INTERVAL_MINUTES
        progress = min(t / RAMP_MINUTES, 1.0)
        simulated_rate = round(pre_rate + (post_rate - pre_rate) * progress, 2)
        _log(audit_log, "CHECKPOINT",
             f"t+{t}min: simulated success rate {simulated_rate}% (threshold to resolve: {threshold}%).",
             elapsed_minutes=t, simulated_rate=simulated_rate)

        if simulated_rate >= threshold:
            _log(audit_log, "RESOLVED",
                 f"Success rate reached {simulated_rate}%, within {int(RESOLVED_RATIO*100)}% of baseline "
                 f"({baseline_rate}%). Intervention stopped automatically.")
            return {"status": "RESOLVED", "resolution_time_minutes": t, "final_rate": simulated_rate}

    # Ran out of time without reaching the resolved threshold
    final_rate = round(pre_rate + (post_rate - pre_rate) * min(MAX_INTERVENTION_MINUTES / RAMP_MINUTES, 1.0), 2)
    _log(audit_log, "MAX_DURATION_REACHED",
         f"Reached the {MAX_INTERVENTION_MINUTES}-minute cap without returning to baseline "
         f"(rate at cutoff: {final_rate}%). Escalating for continued human monitoring.")
    return {"status": "PARTIAL_RECOVERY", "resolution_time_minutes": MAX_INTERVENTION_MINUTES, "final_rate": final_rate}


# ---------------------------------------------------------------------------
# 5. Top-level orchestration
# ---------------------------------------------------------------------------
def _make_incident_id(incident: dict, segment: dict) -> str:
    ts = pd.Timestamp(incident["current_window_start"]).strftime("%Y%m%dT%H%M%S")
    seg_part = "-".join(str(v) for v in segment.values()) if segment else "PORTFOLIO"
    return f"INC-{ts}-{seg_part}"


def respond_to_incident(investigation: dict, incident: dict, df: pd.DataFrame,
                         human_approved: bool = False) -> dict:
    """
    Main entry point. `investigation` is the dict returned by
    investigator.investigate() (including its internal `_evidence` key —
    firefighter.py is meant to consume the full pipeline object, not the
    stripped-down display JSON). `incident` is the detector's output that
    produced that investigation. `df` is the raw transaction dataframe.

    `human_approved` simulates a human clicking "approve" on a
    PENDING_APPROVAL action from a prior call — in production this would
    be a second call into this same function after a human reviewed step 3's
    output.
    """
    audit_log = []
    evidence = investigation.get("_evidence", {})
    segment = evidence.get("segment", {})
    incident_id = _make_incident_id(incident, segment)

    _log(audit_log, "INCIDENT_RECEIVED",
         f"Firefighter received incident {incident_id} (severity={investigation.get('severity')}, "
         f"confidence={investigation.get('confidence_score')}).")

    if not incident.get("incident_detected"):
        _log(audit_log, "NO_ACTION", "No active incident — nothing to respond to.")
        return {
            "incident_id": incident_id,
            "status": "NO_INCIDENT",
            "intervention": "MONITOR",
            "intervention_status": "NOT_REQUIRED",
            "baseline_success_rate": None,
            "pre_intervention_success_rate": None,
            "post_intervention_success_rate": None,
            "revenue_at_risk": 0,
            "revenue_protected": 0,
            "transactions_affected": 0,
            "transactions_recovered": 0,
            "resolution_time_minutes": None,
            "audit_log": audit_log,
        }

    # --- Guardrail check ---
    guardrails = evaluate_guardrails(investigation)
    for reason in guardrails["blocking_reasons"]:
        _log(audit_log, "GUARDRAIL_BLOCK", reason)

    intervention = select_intervention(investigation)
    _log(audit_log, "INTERVENTION_SELECTED",
         f"Selected '{intervention}' based on dominant failure reason evidence.",
         intervention=intervention)

    seg_analysis = incident.get("segment_analysis") or {}
    baseline_rate = seg_analysis.get("segment_baseline_success_rate", incident.get("baseline_success_rate"))
    pre_rate = seg_analysis.get("segment_current_success_rate", incident.get("current_success_rate"))
    transactions_affected = seg_analysis.get("segment_failed_txns", incident.get("affected_transactions"))
    revenue_at_risk = investigation.get("revenue_at_risk", incident.get("revenue_at_risk"))

    can_execute = guardrails["can_auto_execute"] or human_approved

    if not can_execute:
        _log(audit_log, "PENDING_APPROVAL",
             "Guardrails require human sign-off before this intervention can be simulated/executed.")
        return {
            "incident_id": incident_id,
            "status": "PENDING_APPROVAL",
            "intervention": intervention,
            "intervention_status": "PENDING_APPROVAL",
            "baseline_success_rate": baseline_rate,
            "pre_intervention_success_rate": pre_rate,
            "post_intervention_success_rate": None,
            "revenue_at_risk": revenue_at_risk,
            "revenue_protected": 0,
            "transactions_affected": transactions_affected,
            "transactions_recovered": 0,
            "resolution_time_minutes": None,
            "audit_log": audit_log,
        }

    if human_approved and not guardrails["can_auto_execute"]:
        _log(audit_log, "HUMAN_APPROVED_OVERRIDE",
             "A human approved this action despite guardrails blocking auto-execution. Proceeding.")

    # --- MONITOR / ESCALATE_TO_HUMAN: no active fix, nothing to simulate ---
    if intervention in ("MONITOR", "ESCALATE_TO_HUMAN"):
        _log(audit_log, "NO_AUTOMATED_FIX",
             f"'{intervention}' involves no automated traffic change; observing only.")
        return {
            "incident_id": incident_id,
            "status": "MONITORING" if intervention == "MONITOR" else "ESCALATED",
            "intervention": intervention,
            "intervention_status": "EXECUTED",
            "baseline_success_rate": baseline_rate,
            "pre_intervention_success_rate": pre_rate,
            "post_intervention_success_rate": pre_rate,
            "revenue_at_risk": revenue_at_risk,
            "revenue_protected": 0,
            "transactions_affected": transactions_affected,
            "transactions_recovered": 0,
            "resolution_time_minutes": None,
            "audit_log": audit_log,
        }

    # --- Active interventions: REDIRECT_TRAFFIC / REDUCE_AFFECTED_ROUTE / ROLLBACK ---
    dim, affected_value, sibling_avg_rate = pick_recovery_dimension(evidence)
    if sibling_avg_rate is None:
        _log(audit_log, "INSUFFICIENT_EVIDENCE",
             "No usable sibling comparison data to model a recovery target; escalating instead.")
        return {
            "incident_id": incident_id,
            "status": "ESCALATED",
            "intervention": "ESCALATE_TO_HUMAN",
            "intervention_status": "EXECUTED",
            "baseline_success_rate": baseline_rate,
            "pre_intervention_success_rate": pre_rate,
            "post_intervention_success_rate": pre_rate,
            "revenue_at_risk": revenue_at_risk,
            "revenue_protected": 0,
            "transactions_affected": transactions_affected,
            "transactions_recovered": 0,
            "resolution_time_minutes": None,
            "audit_log": audit_log,
        }

    effectiveness = INTERVENTION_EFFECTIVENESS[intervention]
    target_rate = min(pre_rate + effectiveness * (sibling_avg_rate - pre_rate), sibling_avg_rate)
    target_rate = round(max(target_rate, pre_rate), 2)

    _log(audit_log, "SIMULATION_TARGET_SET",
         f"Modeling recovery toward '{dim}' peer average ({sibling_avg_rate}%) at "
         f"{int(effectiveness*100)}% assumed effectiveness for {intervention} -> target {target_rate}%.",
         recovery_dimension=dim, sibling_avg_rate=sibling_avg_rate, effectiveness=effectiveness)

    segment_df = get_segment_window_df(df, incident, segment)
    sim = simulate_intervention(segment_df, target_rate)
    post_rate = sim["achieved_rate"]

    _log(audit_log, "INTERVENTION_EXECUTED",
         f"SIMULATED {intervention} on {describe_segment(segment)}: {len(sim['recovered_ids'])} of "
         f"{transactions_affected} previously-failed transactions modeled as recovered, "
         f"projected rate {pre_rate}% -> {post_rate}%.",
         transactions_recovered=len(sim["recovered_ids"]), recovered_amount=sim["recovered_amount"])

    checkpoint_result = run_checkpoints(pre_rate, post_rate, baseline_rate, audit_log)

    status = checkpoint_result["status"]
    intervention_status = "ROLLED_BACK" if status == "ROLLED_BACK" else "EXECUTED"

    # If we rolled back, no revenue is actually protected — the fix didn't work.
    transactions_recovered = 0 if status == "ROLLED_BACK" else len(sim["recovered_ids"])
    revenue_protected = 0 if status == "ROLLED_BACK" else sim["recovered_amount"]

    _log(audit_log, "FINAL_STATUS", f"Incident {incident_id} closed with status '{status}'.")

    return {
        "incident_id": incident_id,
        "status": status,
        "intervention": intervention,
        "intervention_status": intervention_status,
        "baseline_success_rate": baseline_rate,
        "pre_intervention_success_rate": pre_rate,
        "post_intervention_success_rate": post_rate,  # SIMULATED
        "revenue_at_risk": revenue_at_risk,            # ACTUAL (observed failed GMV)
        "revenue_protected": revenue_protected,        # SIMULATED (sum of recovered txns' real amounts)
        "transactions_affected": transactions_affected,  # ACTUAL
        "transactions_recovered": transactions_recovered,  # SIMULATED
        "resolution_time_minutes": checkpoint_result["resolution_time_minutes"],
        "audit_log": audit_log,
    }


# ---------------------------------------------------------------------------
# 6. End-to-end demo: DATA -> DETECTOR -> INVESTIGATOR -> FIREFIGHTER
# ---------------------------------------------------------------------------
def print_full_report(incident: dict, investigation: dict, result: dict) -> None:
    print("\n" + "🔥 " + "=" * 56)
    print("🔥 INCIDENT")
    print("🔥 " + "=" * 56)
    print(f"  Detected              : {incident['incident_detected']}")
    print(f"  Severity              : {incident['severity']}")
    print(f"  Window                : {incident['current_window_start']} → {incident['current_window_end']}")
    print(f"  Baseline / current     : {incident['baseline_success_rate']}% / {incident['current_success_rate']}%")
    print(f"  Portfolio revenue at risk (window): ₹{incident['revenue_at_risk']:,.2f}")

    print("\n" + "🧠 " + "=" * 56)
    print("🧠 DIAGNOSIS")
    print("🧠 " + "=" * 56)
    print(f"  Summary   : {investigation['incident_summary']}")
    print(f"  Root cause: {investigation['probable_root_cause']}")
    print(f"  Confidence: {investigation['confidence_score']}/100")
    print(f"  Segments  : {investigation['affected_segments']}")

    print("\n" + "🚒 " + "=" * 56)
    print("🚒 INTERVENTION")
    print("🚒 " + "=" * 56)
    print(f"  Incident ID          : {result['incident_id']}")
    print(f"  Status               : {result['status']}")
    print(f"  Intervention         : {result['intervention']}")
    print(f"  Intervention status  : {result['intervention_status']}")

    print("\n" + "📈 " + "=" * 56)
    print("📈 BEFORE vs AFTER  (pre = ACTUAL, post = SIMULATED)")
    print("📈 " + "=" * 56)
    print(f"  Baseline success rate         : {result['baseline_success_rate']}%")
    print(f"  Pre-intervention success rate : {result['pre_intervention_success_rate']}%  [ACTUAL]")
    print(f"  Post-intervention success rate: {result['post_intervention_success_rate']}%  [SIMULATED]")
    print(f"  Transactions affected          : {result['transactions_affected']}  [ACTUAL]")
    print(f"  Transactions recovered (model) : {result['transactions_recovered']}  [SIMULATED]")
    print(f"  Resolution time                : {result['resolution_time_minutes']} min")

    print("\n" + "💰 " + "=" * 56)
    print("💰 REVENUE PROTECTED")
    print("💰 " + "=" * 56)
    print(f"  Revenue at risk (observed)  : ₹{result['revenue_at_risk']:,.2f}  [ACTUAL]")
    print(f"  Revenue protected (modeled): ₹{result['revenue_protected']:,.2f}  [SIMULATED]")

    print("\n" + "📜 " + "=" * 56)
    print("📜 AUDIT TRAIL")
    print("📜 " + "=" * 56)
    for entry in result["audit_log"]:
        print(f"  [{entry['timestamp']}] {entry['action']}: {entry['reason']}")


if __name__ == "__main__":
    data = load_data(CSV_PATH)
    incident = detect_incident(data)
    investigation = investigate(incident, data)

    # First pass: no human approval yet. Because our injected incident is
    # CRITICAL severity, the guardrails will block auto-execution here.
    pending_result = respond_to_incident(investigation, incident, data, human_approved=False)
    print("\n" + "#" * 60)
    print("# PASS 1 — before human approval")
    print("#" * 60)
    print(json.dumps({k: v for k, v in pending_result.items()}, indent=2, default=str))

    # Second pass: simulate a human reviewing the diagnosis and approving
    # the recommended action (this is the realistic path for a
    # CRITICAL/HIGH-severity incident under these guardrails).
    approved_result = respond_to_incident(investigation, incident, data, human_approved=True)

    print_full_report(incident, investigation, approved_result)

    print("\n--- Firefighter result (JSON) ---")
    print(json.dumps(approved_result, indent=2, default=str))
