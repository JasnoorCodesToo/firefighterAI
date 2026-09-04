# 🔥 Financial Firefighter

**AI-powered incident response for payment infrastructure.**

Financial Firefighter is a fintech incident-response simulation. It watches a stream of
payment transactions, detects when the payment success rate suddenly drops, figures out
*why*, recommends a fix, and — after a human signs off when the situation calls for it —
simulates that fix and measures whether it actually worked.

> ⚠️ **This is a demo/simulation.** It does not connect to, route, or affect any real
> payment infrastructure or real money. All "interventions" and "recovery" numbers are
> modeled from historical transaction data. See [Safety Disclaimer](#safety-disclaimer).

---

## The problem

A payment platform normally has a stable, healthy success rate. When it suddenly drops —
say, one bank's UPI route starts timing out — every minute that goes unnoticed is lost
revenue and a worse customer experience. Payments teams need to:

1. Notice the drop fast, even if it's hiding inside a healthy-looking portfolio average.
2. Figure out *where* it's coming from (which bank, method, device, city) and *why*.
3. Decide on a fix without guessing, and get sign-off before anything risky happens.
4. Confirm the fix worked — and undo it automatically if it didn't.
5. Know exactly how much revenue was protected, and have a clean audit trail after.

## The solution

Financial Firefighter automates that loop end-to-end as a statistical pipeline — no
hardcoded assumptions about which bank or payment method is "the usual suspect." It finds
the anomaly purely by comparing current behavior against each segment's own history.

## Architecture

```
CSV (transaction log)
        │
        ▼
   ANALYSIS          analyze.py     — load data, compute health metrics
        │
        ▼
   DETECTOR           detector.py    — find the incident + likely root-cause segment
        │
        ▼
   INVESTIGATOR        investigator.py — gather evidence, diagnose, score confidence
        │
        ▼
   HUMAN APPROVAL      (required for low-confidence or HIGH/CRITICAL incidents)
        │
        ▼
   FIREFIGHTER         firefighter.py — simulate the intervention, checkpoint, recover/rollback
        │
        ▼
   AUDIT TRAIL         every decision and checkpoint logged
```

`app.py` is a Streamlit dashboard that sits on top of this pipeline and drives it
end-to-end — it never bypasses any of the logic below.

## Modules

| File | Role |
|---|---|
| `analyze.py` | Loads the transaction CSV, computes overall success rate, GMV, failed GMV, and breakdowns by payment method / bank / time window. |
| `detector.py` | Compares the current trailing time window against a historical baseline (relative drop % *and* z-score, so a single noisy window doesn't false-alarm), classifies severity, and drills into `payment_method` / `bank` / `device` / `city` to find the specific segment that's actually on fire. |
| `investigator.py` | Turns the detector's output into a full diagnosis: sibling comparisons ("HDFC vs every other bank"), dominant failure reason, device pattern, onset timing, a 0–100 confidence score, and a recommended action. All numbers are computed in plain Python/pandas — see [AI investigation](#ai-investigation) below. |
| `firefighter.py` | Applies safety guardrails, decides whether human approval is required, selects a bounded intervention, **simulates** it against real transaction rows, runs timed recovery checkpoints, rolls back if there's no improvement, and maintains the audit log. |
| `run_demo.py` | End-to-end backend smoke test (Steps 1–2) with sanity assertions. |
| `app.py` | Streamlit dashboard — the operations command center UI. |
| `financial_firefighter_transactions.csv` | Synthetic transaction dataset used for the demo. |

## AI investigation

`investigator.py` deliberately separates **numbers** from **narrative**:

- Every number in the output (revenue at risk, confidence score, segment comparisons) is
  computed directly from the raw transaction data in Python — the LLM never invents or
  touches a statistic.
- If an `ANTHROPIC_API_KEY` environment variable is set, an LLM call turns the
  already-computed evidence into readable prose, under a prompt that explicitly forbids
  introducing new numbers.
- If no API key is available (or the call fails for any reason), a deterministic
  template-based fallback produces equivalent — if less fluent — prose. **The pipeline
  always runs end-to-end with zero external dependencies.**

## Human-in-the-loop approval

Nothing risky executes without a human clicking approve. `investigator.py` flags
`human_approval_required=True` whenever confidence is below the threshold *or* severity is
HIGH/CRITICAL. `firefighter.py` independently re-checks the same gates before touching
anything (defense in depth) and will only proceed on a blocked incident if
`human_approved=True` is explicitly passed in — exactly what the dashboard's
**APPROVE & EXECUTE RESPONSE** button does.

## Safety guardrails

- Confidence score below **75** → never auto-executes.
- Severity **HIGH** or **CRITICAL** → never auto-executes.
- Maximum intervention duration: **30 minutes**.
- No improvement observed at the first checkpoint → automatic **rollback**.
- Success rate recovers to within 95% of baseline → automatically marked **resolved**.
- These checks are re-verified inside `firefighter.py` itself — the Streamlit UI cannot
  bypass them; it can only supply the `human_approved` flag the backend already expects.

## Simulated intervention & rollback

When an intervention is approved, `firefighter.py` builds a synthetic "after" copy of the
affected segment's transactions: previously-failed transactions are deterministically
"recovered" (earliest failures first) up to a modeled target success rate, based on how
much of the gap to a healthy sibling segment's average the chosen intervention is assumed
to close. The result is checkpointed every 5 simulated minutes up to the 30-minute cap:

- If the modeled rate never improves on the pre-intervention rate, it's **rolled back**
  immediately and revenue protected is reported as **₹0**.
- If it reaches within 95% of baseline, it's marked **resolved** early.
- If it's still improving but hasn't fully recovered by the cap, it's **partial recovery**.

## Revenue protection & audit trail

Revenue *at risk* is always an **actual, observed** number (real failed GMV in the
incident window). Revenue *protected* is a **simulated/modeled** number — it is always 0
on rollback, and is clearly labeled as modeled everywhere it's shown, including in the
dashboard. Every decision the firefighter makes — guardrail checks, intervention
selection, checkpoints, rollback/resolution — is appended to a structured audit log with a
timestamp and reason, which the dashboard renders as a full audit trail.

## Streamlit dashboard

`app.py` is a dark, fintech-style operations console built entirely on top of the existing
backend (it imports and calls `analyze.py` / `detector.py` / `investigator.py` /
`firefighter.py` directly — it does not reimplement any logic). It shows:

- KPI cards: success rate, money at risk, affected transactions, severity
- Incident Command panel with the live detected incident
- AI Investigation panel: root cause, supporting evidence, confidence, recommended action
- Response panel with the human approval gate and **APPROVE & EXECUTE RESPONSE** button
- Before/after recovery chart, revenue protected, transactions recovered, resolution time
- Full audit trail
- Sidebar: system status, data source, transactions monitored, detection window,
  confidence threshold, auto-execution policy, and a **🚨 SIMULATE PAYMENT FIRE** button
  that re-runs the real pipeline (nothing about the incident is hardcoded)

## Setup instructions

**Requirements:** Python 3.10+

```bash
# from inside the project folder
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Optional — to enable LLM-generated narrative text instead of the deterministic fallback:

```bash
export ANTHROPIC_API_KEY=your-key-here   # Windows (PowerShell): $env:ANTHROPIC_API_KEY="your-key-here"
```

The CSV path is resolved automatically: it looks for
`financial_firefighter_transactions.csv` next to the Python files, or you can override it
explicitly with `FIREFIGHTER_CSV_PATH`.

## Demo instructions

**Backend only (no browser):**

```bash
python run_demo.py
```

This runs Steps 1–2 end-to-end with sanity assertions and prints the full incident object.
You can also run each module standalone (`python detector.py`, `python investigator.py`,
`python firefighter.py`) to see that stage's output in isolation.

**Full dashboard:**

```bash
streamlit run app.py
```

Opens in your browser. The pipeline runs automatically on load; click
**🚨 SIMULATE PAYMENT FIRE** in the sidebar to reset and re-run it, and
**APPROVE & EXECUTE RESPONSE** to approve a pending intervention and watch the
before/after recovery, revenue protected, and audit trail populate.

## Safety disclaimer

Financial Firefighter is a **simulation and demo project only**. It does not connect to
any real payment gateway, bank, or routing system, and does not move real money. All
"executed" interventions are constructed by modeling a synthetic post-intervention copy of
historical transaction rows — nothing here can affect live traffic. It is intended to
demonstrate incident-response architecture, human-in-the-loop AI decision-making, and
safety-guardrail design, not to be deployed against production payment infrastructure as-is.
