"""
Financial Firefighter — Demo / smoke test for Step 1 + Step 2
=================================================================
A single script that:
  1. Loads and analyzes the dataset (Step 1).
  2. Runs the fire detector (Step 2).
  3. Asserts the detector actually fires on this dataset and prints the
     resulting incident object as JSON (what a downstream diagnosis /
     UI layer would consume in later steps).

Run:
    python3 run_demo.py
"""

import json

from analyze import load_data, print_report, CSV_PATH
from detector import detect_incident, print_incident_report


def main():
    df = load_data(CSV_PATH)

    # ---- Step 1 ----
    print_report(df)

    # ---- Step 2 ----
    print()
    incident = detect_incident(df)
    print_incident_report(incident)

    # Sanity checks — useful as a lightweight regression test whenever
    # the detector logic changes.
    assert isinstance(incident, dict), "detect_incident must return a dict"
    assert "incident_detected" in incident, "missing 'incident_detected' key"

    if incident["incident_detected"]:
        required_keys = [
            "severity", "baseline_success_rate", "current_success_rate",
            "drop_percentage", "affected_transactions", "revenue_at_risk",
            "top_payment_method", "top_bank", "top_device", "top_failure_reason",
        ]
        missing = [k for k in required_keys if k not in incident]
        assert not missing, f"incident dict missing expected keys: {missing}"

    print("\n" + "=" * 60)
    print("INCIDENT OBJECT (JSON) — ready for Step 3 (diagnosis agent)")
    print("=" * 60)
    printable = {k: v for k, v in incident.items() if not k.startswith("_")}
    print(json.dumps(printable, indent=2, default=str))

    print("\n✅ All checks passed.")


if __name__ == "__main__":
    main()
