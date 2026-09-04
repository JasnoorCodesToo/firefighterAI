"""
FINANCIAL FIREFIGHTER
AI-powered payment incident response

Step 5 — Streamlit operations command center.

This UI deliberately reuses the existing Step 1–4 backend:
    analyze.py -> detector.py -> investigator.py -> firefighter.py

No real payment routing or financial action is performed. Firefighter
results are simulations derived by the existing backend.
"""

from pathlib import Path
import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analyze import load_data, overall_metrics, CSV_PATH
from detector import detect_incident
from investigator import investigate
from firefighter import respond_to_incident


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Financial Firefighter",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling — dark fintech operations console
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        .stApp {
            background: #090d12;
            color: #eef2f7;
        }

        [data-testid="stHeader"] {
            background: #090d12;
        }

        [data-testid="stSidebar"] {
            background: #0d131b;
            border-right: 1px solid #1e2935;
        }

        .block-container {
            max-width: 1500px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            background: linear-gradient(135deg, #101923 0%, #0b1118 100%);
            border: 1px solid #253342;
            border-radius: 18px;
            padding: 26px 30px;
            margin-bottom: 20px;
        }

        .hero-title {
            font-size: 2.15rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin: 0;
        }

        .hero-subtitle {
            color: #98a6b7;
            margin-top: 5px;
            font-size: 1rem;
        }

        .status-pill {
            display: inline-block;
            margin-top: 15px;
            padding: 7px 13px;
            border-radius: 999px;
            background: #10261e;
            border: 1px solid #24513d;
            color: #75d8aa;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
        }

        .kpi {
            background: #101720;
            border: 1px solid #202c39;
            border-radius: 15px;
            padding: 18px 19px;
            min-height: 120px;
        }

        .kpi-label {
            color: #8e9baa;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }

        .kpi-value {
            color: #f4f7fb;
            font-size: 1.65rem;
            font-weight: 800;
            margin-top: 8px;
        }

        .kpi-sub {
            color: #718091;
            font-size: 0.78rem;
            margin-top: 3px;
        }

        .section {
            color: #f4f7fb;
            font-size: 1.18rem;
            font-weight: 800;
            margin: 27px 0 12px;
        }

        .incident {
            background: #171216;
            border: 1px solid #5b3034;
            border-left: 5px solid #e46b70;
            border-radius: 15px;
            padding: 21px 23px;
            margin: 8px 0 14px;
        }

        .healthy {
            background: #0f1b17;
            border: 1px solid #214b3b;
            border-left: 5px solid #51c996;
            border-radius: 15px;
            padding: 21px 23px;
            margin: 8px 0 14px;
        }

        .panel {
            background: #101720;
            border: 1px solid #202c39;
            border-radius: 15px;
            padding: 20px 22px;
            height: 100%;
        }

        .panel-title {
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 13px;
        }

        .muted {
            color: #8e9baa;
        }

        .money-risk {
            font-size: 2.1rem;
            font-weight: 850;
            letter-spacing: -0.03em;
        }

        .approval {
            background: #201b10;
            border: 1px solid #6b5423;
            border-radius: 13px;
            padding: 16px 18px;
            margin: 14px 0;
        }

        .simulation-note {
            background: #111923;
            border: 1px dashed #344557;
            border-radius: 10px;
            padding: 11px 14px;
            color: #aeb9c7;
            font-size: 0.82rem;
            margin: 10px 0;
        }

        .rollback {
            background: #211514;
            border: 1px solid #6a3530;
            border-radius: 14px;
            padding: 18px;
            margin-top: 14px;
        }

        .audit-row {
            background: #0d141c;
            border: 1px solid #1d2935;
            border-radius: 9px;
            padding: 10px 13px;
            margin: 6px 0;
        }

        .audit-event {
            font-weight: 750;
            color: #e8edf4;
        }

        .audit-message {
            color: #91a0b0;
            font-size: 0.84rem;
            margin-top: 3px;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 10px;
            font-weight: 750;
            min-height: 44px;
        }

        [data-testid="stMetric"] {
            background: #101720;
            border: 1px solid #202c39;
            border-radius: 12px;
            padding: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data / pipeline helpers
# ---------------------------------------------------------------------------

def resolve_csv_path() -> str:
    """Prefer a CSV beside the project files; fall back to the backend path."""
    local_path = Path(__file__).resolve().parent / Path(CSV_PATH).name
    if local_path.exists():
        return str(local_path)
    return CSV_PATH


@st.cache_data(show_spinner=False)
def load_dashboard_data(csv_path: str) -> pd.DataFrame:
    return load_data(csv_path)


def run_detection_pipeline():
    """Run Steps 1–3 once and return their real outputs."""
    csv_path = resolve_csv_path()
    df = load_dashboard_data(csv_path)
    incident = detect_incident(df)
    investigation = investigate(incident, df)
    return df, incident, investigation


def reset_demo():
    for key in (
        "demo_started",
        "incident",
        "investigation",
        "pending_result",
        "firefighter_result",
    ):
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def money(value) -> str:
    if value is None:
        return "₹0.00"
    try:
        return f"₹{float(value):,.2f}"
    except (TypeError, ValueError):
        return "₹0.00"


def pct(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "—"


def safe_text(value) -> str:
    if value is None:
        return "—"
    return html.escape(str(value))


def severity_label(severity: str) -> str:
    return severity if severity and severity != "NONE" else "HEALTHY"


def format_segment(segments) -> str:
    if not segments:
        return "Portfolio-wide"
    return " • ".join(str(x) for x in segments)


def render_audit_log(audit_log):
    if not audit_log:
        st.info("No audit events returned by the firefighter.")
        return

    for entry in audit_log:
        if isinstance(entry, dict):
            event = entry.get("event") or entry.get("type") or entry.get("action") or "EVENT"
            message = entry.get("message") or entry.get("detail") or ""
            timestamp = entry.get("timestamp") or entry.get("time") or ""
        else:
            event = "EVENT"
            message = str(entry)
            timestamp = ""

        label = f"{timestamp}  " if timestamp else ""
        st.markdown(
            f"""
            <div class="audit-row">
                <div class="audit-event">{safe_text(label)}{safe_text(event)}</div>
                <div class="audit-message">{safe_text(message)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def make_before_after_chart(pre_rate, post_rate, baseline_rate):
    labels = ["Pre-intervention", "Post-intervention"]
    values = [pre_rate, post_rate]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            text=[pct(pre_rate), pct(post_rate)],
            textposition="auto",
        )
    )

    if baseline_rate is not None:
        fig.add_hline(
            y=baseline_rate,
            line_dash="dash",
            annotation_text=f"Baseline {pct(baseline_rate)}",
            annotation_position="top left",
        )

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis_title="Success rate (%)",
        yaxis=dict(range=[0, 100]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d9e1ea"),
    )
    return fig


# ---------------------------------------------------------------------------
# Initialize pipeline
# ---------------------------------------------------------------------------

if "demo_started" not in st.session_state:
    st.session_state.demo_started = True

try:
    df, incident, investigation = run_detection_pipeline()
    st.session_state.incident = incident
    st.session_state.investigation = investigation
except Exception as exc:
    st.error(f"Unable to load the Financial Firefighter pipeline: {exc}")
    st.stop()

incident = st.session_state.incident
investigation = st.session_state.investigation

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🔥 FINANCIAL FIREFIGHTER</div>
        <div class="hero-subtitle">AI-powered payment incident response</div>
        <div class="status-pill">● MONITORING</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

metrics = overall_metrics(df)

with st.sidebar:
    st.markdown("## Operations")
    st.caption("Financial Firefighter control center")

    if st.button("🚨 SIMULATE PAYMENT FIRE", use_container_width=True):
        reset_demo()
        st.session_state.demo_started = True
        st.rerun()

    st.divider()
    st.markdown("### System Status")
    st.success("MONITORING")

    st.markdown("### Data Source")
    st.caption(Path(resolve_csv_path()).name)

    st.markdown("### Transactions Monitored")
    st.write(f"{metrics['total_transactions']:,}")

    st.markdown("### Detection Window")
    st.write(str(incident.get("window_size", "15min")))

    # Read the actual threshold from investigator.py when available.
    try:
        import investigator as investigator_module
        confidence_threshold = getattr(
            investigator_module,
            "CONFIDENCE_HUMAN_APPROVAL_THRESHOLD",
            75,
        )
    except Exception:
        confidence_threshold = 75

    st.markdown("### AI Confidence Threshold")
    st.write(f"{confidence_threshold}/100")

    st.markdown("### Auto-execution Policy")
    st.write("HIGH / CRITICAL → Human approval")

    st.divider()
    st.caption("Simulation only — no real payment routing is executed.")


# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

baseline_rate = incident.get("baseline_success_rate")
current_rate = incident.get("current_success_rate")
severity = severity_label(incident.get("severity"))
revenue_at_risk = investigation.get(
    "revenue_at_risk",
    incident.get("revenue_at_risk", 0),
)
affected_transactions = incident.get("affected_transactions", 0)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">Payment Success Rate</div>
            <div class="kpi-value">{pct(current_rate)}</div>
            <div class="kpi-sub">Baseline {pct(baseline_rate)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">Money at Risk</div>
            <div class="kpi-value">{money(revenue_at_risk)}</div>
            <div class="kpi-sub">Observed failed GMV in incident window</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">Affected Transactions</div>
            <div class="kpi-value">{int(affected_transactions):,}</div>
            <div class="kpi-sub">Current incident window</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">Incident Severity</div>
            <div class="kpi-value">{safe_text(severity)}</div>
            <div class="kpi-sub">Detector classification</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Incident panel
# ---------------------------------------------------------------------------

st.markdown('<div class="section">Incident Command</div>', unsafe_allow_html=True)

if not incident.get("incident_detected"):
    st.markdown(
        """
        <div class="healthy">
            <strong>✓ SYSTEM HEALTHY</strong><br>
            <span class="muted">
                No active payment incident was detected by the existing detector.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    segments = format_segment(investigation.get("affected_segments"))

    st.markdown(
        f"""
        <div class="incident">
            <div style="font-size:1.1rem;font-weight:850;">🔥 PAYMENT INCIDENT DETECTED</div>
            <div class="muted" style="margin-top:8px;">
                Success rate has moved from
                <strong>{pct(baseline_rate)}</strong>
                to
                <strong>{pct(current_rate)}</strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Affected Segment", segments)
    with c2:
        st.metric("Investigation Confidence", f"{investigation.get('confidence_score', 0)}/100")
    with c3:
        st.metric("Money at Risk", money(revenue_at_risk))


# ---------------------------------------------------------------------------
# Investigation + response
# ---------------------------------------------------------------------------

left, right = st.columns([1.15, 0.85])

with left:
    st.markdown('<div class="section">🧠 AI Investigation</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown('<div class="panel-title">Root Cause</div>', unsafe_allow_html=True)
    st.write(investigation.get("probable_root_cause") or "No root cause — system is healthy.")

    st.markdown('<div class="panel-title">Supporting Evidence</div>', unsafe_allow_html=True)
    evidence = investigation.get("supporting_evidence", [])
    if evidence:
        for item in evidence:
            if isinstance(item, dict):
                st.write("• " + " — ".join(f"{k}: {v}" for k, v in item.items()))
            else:
                st.write(f"• {item}")
    else:
        st.write("No active evidence.")

    st.markdown('<div class="panel-title">Affected Segments</div>', unsafe_allow_html=True)
    st.write(investigation.get("affected_segments") or "None")

    st.markdown('<div class="panel-title">Confidence</div>', unsafe_allow_html=True)
    st.progress(min(max(int(investigation.get("confidence_score", 0)), 0), 100) / 100)
    st.write(f"{investigation.get('confidence_score', 0)}/100")

    st.markdown('<div class="panel-title">Recommended Action</div>', unsafe_allow_html=True)
    st.info(investigation.get("recommended_action", "No action needed."))

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="section">🚒 Response</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    if not incident.get("incident_detected"):
        st.success("No response required.")
    else:
        # First firefighter call: backend decides whether approval is needed.
        if "pending_result" not in st.session_state:
            st.session_state.pending_result = respond_to_incident(
                investigation,
                incident,
                df,
                human_approved=False,
            )

        pending = st.session_state.pending_result

        st.markdown("**Recommended intervention**")
        st.code(str(pending.get("intervention", "MONITOR")), language=None)

        if pending.get("status") == "PENDING_APPROVAL":
            st.markdown(
                """
                <div class="approval">
                    <strong>⚠️ HUMAN APPROVAL REQUIRED</strong><br>
                    <span class="muted">
                        The existing safety guardrails have blocked automatic execution.
                        A human must explicitly approve the simulated response.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                "APPROVE & EXECUTE RESPONSE",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.firefighter_result = respond_to_incident(
                    investigation,
                    incident,
                    df,
                    human_approved=True,
                )
                st.rerun()
        else:
            st.success(f"Response status: {pending.get('status')}")

        st.markdown(
            """
            <div class="simulation-note">
                SIMULATION: firefighter.py models a bounded response using the
                existing transaction data. No real payment route is changed.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Firefighter result / before-after
# ---------------------------------------------------------------------------

result = st.session_state.get("firefighter_result")

if result:
    st.markdown('<div class="section">Recovery Outcome</div>', unsafe_allow_html=True)

    if result.get("status") == "ROLLED_BACK":
        st.markdown(
            """
            <div class="rollback">
                <strong>↩️ INTERVENTION ROLLED BACK</strong><br>
                <span class="muted">
                    The existing firefighter checkpoint policy determined that
                    the intervention did not produce sufficient improvement.
                    Revenue protected is therefore reported as ₹0.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    pre_rate = result.get("pre_intervention_success_rate")
    post_rate = result.get("post_intervention_success_rate")
    result_baseline = result.get("baseline_success_rate")

    if post_rate is not None:
        chart_col, stats_col = st.columns([1.5, 1])

        with chart_col:
            st.markdown(
                '<div class="panel"><div class="panel-title">Before vs After</div>',
                unsafe_allow_html=True,
            )
            st.caption("Post-intervention rate is SIMULATED/MODELED.")
            st.plotly_chart(
                make_before_after_chart(pre_rate, post_rate, result_baseline),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with stats_col:
            st.markdown(
                '<div class="panel"><div class="panel-title">Response Impact</div>',
                unsafe_allow_html=True,
            )

            protected = result.get("revenue_protected", 0)
            recovered = result.get("transactions_recovered", 0)
            resolution = result.get("resolution_time_minutes")

            st.markdown(
                f"""
                <div class="kpi" style="margin-bottom:10px;">
                    <div class="kpi-label">Money Protected</div>
                    <div class="money-risk">{money(protected)}</div>
                    <div class="kpi-sub">SIMULATED recovery</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            a, b = st.columns(2)
            with a:
                st.metric("Transactions Recovered", f"{int(recovered):,}")
            with b:
                st.metric(
                    "Resolution Time",
                    f"{resolution} min" if resolution is not None else "—",
                )

            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info(
            f"Response completed with status: {result.get('status', 'UNKNOWN')}. "
            "No post-intervention simulation was returned."
        )


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

st.markdown('<div class="section">Audit Trail</div>', unsafe_allow_html=True)

if result:
    render_audit_log(result.get("audit_log", []))
elif st.session_state.get("pending_result"):
    st.caption("Approval-stage audit events from the firefighter:")
    render_audit_log(st.session_state.pending_result.get("audit_log", []))
else:
    st.info("Run the response flow to populate the audit trail.")

st.caption(
    "Financial Firefighter is a simulation. Actual observed metrics come from "
    "the transaction dataset; post-intervention metrics and revenue protected "
    "are modeled outputs from firefighter.py."
)
