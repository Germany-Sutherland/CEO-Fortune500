# app.py — AutoCEO 2050 (Free‑tier Streamlit Edition)
# ---------------------------------------------------
# One-file app that fuses:
# 1) AutoCEO strategy simulator (scenarios + Monte Carlo KPIs)
# 2) AI Boardroom voting (5 personas + user blend) and transparent ledger
# 3) Agentic teams execution simulator (lightweight progress)
# 4) Leadership FMEA (10 leadership styles, heuristic RPN + ELI5)
# 5) 5W+1H usefulness analysis with CSV/TXT export
#
# Runs on Streamlit Free tier. Dependencies (requirements.txt):
#   streamlit
#   pandas
#   numpy
#   altair
# ---------------------------------------------------

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List
import hashlib
import json
import random
import time

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="AutoCEO 2050 — Agent Boardroom",
    page_icon="🤖",
    layout="wide",
)

# -----------------------------
# Core Data & Structures
# -----------------------------
KPI_NAMES = ["Revenue", "Sustainability", "Resilience", "Reputation", "Compliance"]

STRATEGIES = [
    "Accelerate Product Upgrade",
    "Adjust Global Pricing",
    "Reroute & Decarbonize Supply Chain",
    "Launch Targeted PR/IR Campaign",
    "Co-Develop With Strategic Partner",
    "Open-Source Lite Variant",
    "Pause & Lobby for Regulatory Clarity",
]

AI_PERSONAS: List = [
    ("Finance AI",       {"Revenue": 0.40, "Sustainability": 0.05, "Resilience": 0.15, "Reputation": 0.10, "Compliance": 0.30}),
    ("Sustainability AI", {"Revenue": 0.10, "Sustainability": 0.45, "Resilience": 0.15, "Reputation": 0.20, "Compliance": 0.10}),
    ("Resilience AI",     {"Revenue": 0.15, "Sustainability": 0.20, "Resilience": 0.40, "Reputation": 0.15, "Compliance": 0.10}),
    ("Reputation AI",     {"Revenue": 0.10, "Sustainability": 0.20, "Resilience": 0.10, "Reputation": 0.45, "Compliance": 0.15}),
    ("Compliance AI",     {"Revenue": 0.05, "Sustainability": 0.10, "Resilience": 0.15, "Reputation": 0.10, "Compliance": 0.60}),
]

SCENARIOS = {
    "Geopolitical Disruption": {
        "description": "Tariff shock and port strike in Asia; cross-border payments friction.",
        "volatility": 0.35,
        "kpi_bias": {"Revenue": -0.05, "Sustainability": -0.02, "Resilience": -0.10, "Reputation": 0.00, "Compliance": 0.05},
    },
    "Climate Regulation Shift (EU)": {
        "description": "Stricter carbon caps and recyclability mandates.",
        "volatility": 0.30,
        "kpi_bias": {"Revenue": -0.03, "Sustainability": 0.15, "Resilience": -0.02, "Reputation": 0.05, "Compliance": 0.10},
    },
    "Competitor Breakthrough": {
        "description": "Rival launches next‑gen device; 15% share risk.",
        "volatility": 0.40,
        "kpi_bias": {"Revenue": -0.08, "Sustainability": 0.00, "Resilience": -0.05, "Reputation": -0.05, "Compliance": 0.00},
    },
    "Supply Chain Cyberattack": {
        "description": "Ransomware on Tier‑2 suppliers; data and OT disruption.",
        "volatility": 0.45,
        "kpi_bias": {"Revenue": -0.06, "Sustainability": 0.00, "Resilience": -0.12, "Reputation": -0.02, "Compliance": 0.04},
    },
    "Demand Spike (India)": {
        "description": "+30% demand from India due to incentives.",
        "volatility": 0.25,
        "kpi_bias": {"Revenue": 0.10, "Sustainability": -0.01, "Resilience": -0.03, "Reputation": 0.03, "Compliance": 0.00},
    },
}

STRATEGY_PRIORS = {
    "Accelerate Product Upgrade":       {"Revenue": 0.12, "Sustainability": 0.02, "Resilience": 0.03, "Reputation": 0.06, "Compliance": 0.02},
    "Adjust Global Pricing":            {"Revenue": 0.08, "Sustainability": 0.00, "Resilience": 0.02, "Reputation": -0.02, "Compliance": 0.01},
    "Reroute & Decarbonize Supply Chain": {"Revenue": -0.02, "Sustainability": 0.12, "Resilience": 0.10, "Reputation": 0.05, "Compliance": 0.04},
    "Launch Targeted PR/IR Campaign":   {"Revenue": 0.02, "Sustainability": 0.00, "Resilience": 0.00, "Reputation": 0.10, "Compliance": 0.00},
    "Co-Develop With Strategic Partner": {"Revenue": 0.09, "Sustainability": 0.05, "Resilience": 0.08, "Reputation": 0.07, "Compliance": 0.03},
    "Open-Source Lite Variant":         {"Revenue": -0.03, "Sustainability": 0.06, "Resilience": 0.02, "Reputation": 0.12, "Compliance": 0.01},
    "Pause & Lobby for Regulatory Clarity": {"Revenue": -0.04, "Sustainability": 0.01, "Resilience": 0.04, "Reputation": -0.03, "Compliance": 0.12},
}

@dataclass
class DecisionRecord:
    timestamp: str
    scenario: str
    strategy: str
    kpi_scores: Dict[str, float]
    persona_votes: Dict[str, float]
    seed: int
    prev_hash: str
    hash: str

# -----------------------------
# Session State
# -----------------------------
if "seed" not in st.session_state:
    st.session_state.seed = 42
if "ledger" not in st.session_state:
    st.session_state.ledger: List[DecisionRecord] = []
if "last_decision" not in st.session_state:
    st.session_state.last_decision = None

# -----------------------------
# Helper Functions (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def simulate_strategy_outcomes(seed: int, scenario_key: str, n: int) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    scen = SCENARIOS[scenario_key]
    vol = scen["volatility"]
    bias = scen["kpi_bias"]

    outcomes: Dict[str, pd.DataFrame] = {}
    for strat in STRATEGIES:
        pri = STRATEGY_PRIORS[strat]
        means = np.array([pri[k] + bias.get(k, 0.0) for k in KPI_NAMES])
        stds = np.array([max(0.02, vol * 0.18) for _ in KPI_NAMES])
        sims = rng.normal(loc=means, scale=stds, size=(n, len(KPI_NAMES)))
        df = pd.DataFrame(sims, columns=KPI_NAMES)
        outcomes[strat] = df
    return outcomes

@st.cache_data(show_spinner=False)
def persona_utility(weights: Dict[str, float], kpi_scores: Dict[str, float]) -> float:
    return float(sum(weights[k] * kpi_scores[k] for k in KPI_NAMES))

# -----------------------------
# Layout: Tabs
# -----------------------------
T1, T2, T3, T4, T5 = st.tabs([
    "Dashboard", "AI Boardroom", "Ledger & Agents", "Leadership FMEA", "5W+1H",
])

# -----------------------------
# Tab 1 — Dashboard (Scenario + Simulation)
# -----------------------------
with T1:
    st.subheader("Scenario & Simulation")
    c1, c2, c3 = st.columns([1.6, 1, 1])

    with c1:
        scenario_name = st.selectbox("Scenario", list(SCENARIOS.keys()))
        st.caption(SCENARIOS[scenario_name]["description"])
    with c2:
        rand = st.button("Randomize Seed")
        if rand:
            st.session_state.seed = random.randint(1, 10_000_000)
        st.metric("Seed", st.session_state.seed)
    with c3:
        st.metric("Volatility", f"{SCENARIOS[scenario_name]['volatility']:.2f}")

    st.markdown("User stakeholder priority weights (sum normalized to 1.0)")
    cols = st.columns(5)
    user_w = {}
    defaults = {k: 0.5 for k in KPI_NAMES}
    for i, k in enumerate(KPI_NAMES):
        with cols[i]:
            user_w[k] = st.slider(k, 0.0, 1.0, float(defaults[k]), 0.05)
    s = sum(user_w.values()) or 1.0
    user_w = {k: v / s for k, v in user_w.items()}

    n_sims = st.slider("Simulations", 200, 2000, 800, 100)
    explore = st.slider("Exploration vs Exploitation (epsilon)", 0.0, 1.0, 0.35, 0.05)

    outcomes = simulate_strategy_outcomes(st.session_state.seed, scenario_name, n_sims)
    exp_kpis = {s: df.mean().to_dict() for s, df in outcomes.items()}

    # Heatmap of expected impacts
    heat_df = pd.DataFrame([
        {"Strategy": s, "KPI": k, "Expected Impact": exp_kpis[s][k]} for s in STRATEGIES for k in KPI_NAMES
    ])
    chart = (
        alt.Chart(heat_df)
        .mark_rect()
        .encode(
            x=alt.X("KPI:N", sort=KPI_NAMES),
            y=alt.Y("Strategy:N", sort=STRATEGIES),
            color=alt.Color("Expected Impact:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["Strategy", "KPI", alt.Tooltip("Expected Impact:Q", format=".3f")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)

    # Choose best strategy using AI personas + user blend
    persona_votes: Dict[str, Dict[str, float]] = {s: {} for s in STRATEGIES}
    blended_scores = {}
    for sname in STRATEGIES:
        for persona, w in AI_PERSONAS:
            persona_votes[sname][persona] = persona_utility(w, exp_kpis[sname])
        board_avg = float(np.mean(list(persona_votes[sname].values())))
        user_util = persona_utility(user_w, exp_kpis[sname])
        blended_scores[sname] = 0.7 * board_avg + 0.3 * user_util

    if random.random() < explore:
        best_strategy = random.choice(STRATEGIES)
    else:
        best_strategy = max(blended_scores.items(), key=lambda x: x[1])[0]

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Selected Strategy", best_strategy)
    with m2:
        st.metric("Blend Score", f"{blended_scores[best_strategy]:.3f}")

    # Persona vote table
    votes_tbl = pd.DataFrame(
        {p: [persona_votes[s][p] for s in STRATEGIES] for p, _ in AI_PERSONAS}, index=STRATEGIES
    ).round(3).rename_axis("Strategy").reset_index()
    st.dataframe(votes_tbl, use_container_width=True)

    # KPI trajectory for chosen strategy
    traj_seed = st.session_state.seed + 7
    rng = np.random.default_rng(traj_seed)
    base = np.array([exp_kpis[best_strategy][k] for k in KPI_NAMES])
    noise = rng.normal(0, 0.02, size=(8, len(KPI_NAMES)))
    trajectory = np.maximum(-0.2, np.minimum(0.4, base + np.cumsum(noise, axis=0)))
    traj_df = pd.DataFrame(trajectory, columns=KPI_NAMES)
    traj_df["Week"] = np.arange(1, 9)
    traj_long = traj_df.melt(id_vars=["Week"], var_name="KPI", value_name="Score")

    st.subheader("KPI Trajectory (8 weeks)")
    line = (
        alt.Chart(traj_long)
        .mark_line(point=True)
        .encode(x="Week:O", y=alt.Y("Score:Q", scale=alt.Scale(domain=[-0.2, 0.4])), color="KPI:N",
                tooltip=["Week", "KPI", alt.Tooltip("Score:Q", format=".3f")])
        .properties(height=280)
    )
    st.altair_chart(line, use_container_width=True)

    # Save decision in session for other tabs
    st.session_state.last_decision = {
        "scenario": scenario_name,
        "strategy": best_strategy,
        "expected_kpis": exp_kpis[best_strategy],
        "persona_votes": {p: float(votes_tbl.loc[votes_tbl["Strategy"] == best_strategy, p].values[0]) for p, _ in AI_PERSONAS},
        "blend_score": blended_scores[best_strategy],
        "user_weights": user_w,
        "seed": int(st.session_state.seed),
    }

    # Downloads
    colA, colB = st.columns(2)
    with colA:
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            **st.session_state.last_decision,
        }
        st.download_button(
            label="Download Simulation JSON",
            data=json.dumps(payload, indent=2).encode(),
            file_name="autoceo_simulation.json",
            mime="application/json",
        )
    with colB:
        # Export the whole heat map data
        st.download_button(
            label="Download KPI Grid CSV",
            data=heat_df.to_csv(index=False).encode(),
            file_name="autoceo_kpi_grid.csv",
            mime="text/csv",
        )

# -----------------------------
# Tab 2 — AI Boardroom (Commit to Ledger)
# -----------------------------
with T2:
    st.subheader("AI Boardroom — Commit Decision to Transparent Ledger")

    def _hash_record(payload: dict, prev_hash: str) -> str:
        raw = json.dumps(payload, sort_keys=True).encode() + prev_hash.encode()
        return hashlib.sha256(raw).hexdigest()

    if st.session_state.last_decision is None:
        st.info("Run a simulation in the Dashboard tab to select a strategy.")
    else:
        if st.button("Commit Current Decision to Ledger"):
            ts = datetime.utcnow().isoformat()
            prev_hash = st.session_state.ledger[-1].hash if st.session_state.ledger else "GENESIS"
            payload = {
                "timestamp": ts,
                "scenario": st.session_state.last_decision["scenario"],
                "strategy": st.session_state.last_decision["strategy"],
                "kpi_scores": {k: float(v) for k, v in st.session_state.last_decision["expected_kpis"].items()},
                "persona_votes": st.session_state.last_decision["persona_votes"],
                "seed": st.session_state.last_decision["seed"],
            }
            rec_hash = _hash_record(payload, prev_hash)
            rec = DecisionRecord(
                timestamp=ts,
                scenario=payload["scenario"],
                strategy=payload["strategy"],
                kpi_scores=payload["kpi_scores"],
                persona_votes=payload["persona_votes"],
                seed=payload["seed"],
                prev_hash=prev_hash,
                hash=rec_hash,
            )
            st.session_state.ledger.append(rec)
            st.success("Decision recorded to ledger.")

    if st.session_state.ledger:
        st.markdown("### Transparent Decision Ledger")
        led_df = pd.DataFrame([asdict(x) for x in st.session_state.ledger])
        st.dataframe(led_df, use_container_width=True)
        st.download_button(
            "Export Ledger CSV",
            led_df.to_csv(index=False).encode(),
            "autoceo_ledger.csv",
            "text/csv",
        )
    else:
        st.caption("Ledger is empty.")

# -----------------------------
# Tab 3 — Ledger & Agents (Execution Simulator)
# -----------------------------
with T3:
    st.subheader("Agentic Teams — Execution Simulator")
    st.caption("Lightweight progress visualization for key software agents.")

    AGENTS = [
        ("Contract AI", ["Draft Terms", "Negotiate", "E‑Sign"]),
        ("Logistics AI", ["Reroute", "Optimize Emissions", "Dispatch"]),
        ("R&D AI", ["Design Variant", "Virtual Prototype", "Release to Mfg"]),
        ("Media AI", ["PR Narrative", "Investor Brief", "Localized Posts"]),
    ]

    delay = st.slider("Per‑step delay (seconds)", 0.0, 0.5, 0.10, 0.05, help="Keep small for free tier")

    cols = st.columns(2)
    run_agents = st.button("Run Agents")
    if run_agents:
        for idx, (agent, steps) in enumerate(AGENTS):
            with cols[idx % 2]:
                prog = st.progress(0, text=f"{agent}: starting…")
                for i, step in enumerate(steps, start=1):
                    time.sleep(delay)
                    prog.progress(int(i / len(steps) * 100), text=f"{agent}: {step}")
                prog.progress(100, text=f"{agent}: complete")
        st.success("All agents complete.")

# -----------------------------
# Tab 4 — Leadership FMEA (10 styles)
# -----------------------------
with T4:
    st.subheader("Leadership FMEA — 10 Styles")
    st.caption("Heuristic Failure Mode & Effects Analysis with explainable output.")

    CASES = {
        "": "",
        "Nokia": "Failed to adapt from feature phones to smartphone OS ecosystems.",
        "Kodak": "Underestimated the shift to digital despite inventing it.",
        "Blockbuster": "Late to streaming disruption and subscriptions.",
        "Sears": "Lost retail share to e‑commerce due to slow pivot.",
        "Pan Am": "High fixed costs and deregulation shocks led to collapse.",
    }

    LEADER_STYLES = {
        "Autocratic Leader Agentic AI Agent CEO": "Decides alone, tight control, speed over consensus.",
        "Democratic Leader Agentic AI Agent CEO": "Seeks participation and consensus.",
        "Laissez-Faire Leader Agentic AI Agent CEO": "Hands-off; relies on team autonomy.",
        "Transformational Leader Agentic AI Agent CEO": "Inspires vision and change.",
        "Transactional Leader Agentic AI Agent CEO": "Incentives, KPIs, compliance.",
        "Servant Leader Agentic AI Agent CEO": "People-first, builds trust.",
        "Charismatic Leader Agentic AI Agent CEO": "Storytelling and rallying.",
        "Situational Leader Agentic AI Agent CEO": "Adapts to team maturity.",
        "Visionary Leader Agentic AI Agent CEO": "Long-term bold bets.",
        "Bureaucratic Leader Agentic AI Agent CEO": "Rules and consistency.",
    }

    STYLE_BIASES = {
        "Autocratic":        {"severity": +1, "occurrence": +1, "detection": -1},
        "Democratic":        {"severity":  0, "occurrence": +1, "detection":  0},
        "Laissez-Faire":     {"severity": +1, "occurrence": +2, "detection": -1},
        "Transformational":  {"severity": +2, "occurrence": +1, "detection": -1},
        "Transactional":     {"severity":  0, "occurrence":  0, "detection": +1},
        "Servant":           {"severity":  0, "occurrence":  0, "detection":  0},
        "Charismatic":       {"severity": +2, "occurrence": +1, "detection": -1},
        "Situational":       {"severity": -1, "occurrence": -1, "detection": +1},
        "Visionary":         {"severity": +2, "occurrence": +1, "detection": -1},
        "Bureaucratic":      {"severity": -1, "occurrence":  0, "detection": +2},
    }

    RISKY_KEYWORDS = {
        "merger": (+2, +1, -1), "acquisition": (+2, +1, -1), "layoff": (+2, +2, -1),
        "restructure": (+1, +1, -1), "pivot": (+2, +1, -1), "ai": (+1, +1, -1),
        "cloud": (+1, 0, 0), "shutdown": (+3, +2, -2), "outsourcing": (+1, +1, 0),
        "offshoring": (+1, +1, 0), "automation": (+1, +1, 0), "cybersecurity": (+2, +1, +1),
        "compliance": (+1, 0, +2), "regulation": (+1, 0, +2), "expansion": (+1, +1, -1)
    }

    def clamp(x, lo=1, hi=10):
        return max(lo, min(hi, int(round(x))))

    def base_scores(problem: str, decision: str):
        sev, occ, det = 6, 5, 5
        text = f"{problem} {decision}".lower()
        for kw, (ds, do, dd) in RISKY_KEYWORDS.items():
            if kw in text:
                sev += ds; occ += do; det += dd
        length_factor = min(len(text) // 200, 3)
        sev += length_factor; occ += length_factor
        return clamp(sev), clamp(occ), clamp(det)

    def style_adjusted_scores(sev, occ, det, leader_name: str):
        for key, bias in STYLE_BIASES.items():
            if key in leader_name:
                sev += bias["severity"]; occ += bias["occurrence"]; det += bias["detection"]
                break
        return clamp(sev), clamp(occ), clamp(det)

    def mitigation_for_style(leader_name: str):
        if "Autocratic" in leader_name:
            return ["Create a weekly red‑team review.", "Nominate a devil’s advocate."]
        if "Democratic" in leader_name:
            return ["Timebox discussions.", "Designate final decision owner."]
        if "Laissez-Faire" in leader_name:
            return ["Set biweekly OKRs.", "Install progress dashboards."]
        if "Transformational" in leader_name:
            return ["30‑60‑90 day milestones.", "Risk and dependency registers."]
        if "Transactional" in leader_name:
            return ["Align incentives to long-term value.", "Quarterly KPI audits."]
        if "Servant" in leader_name:
            return ["Balance empathy with performance gates.", "Escalate decisively when risk rises."]
        if "Charismatic" in leader_name:
            return ["Triangulate narrative with data.", "Run pre‑mortems."]
        if "Situational" in leader_name:
            return ["Reassess team readiness each sprint.", "Adapt coaching vs directing."]
        if "Visionary" in leader_name:
            return ["Back‑cast into quarterly deliverables.", "Discovery sprints with kill‑switch gates."]
        if "Bureaucratic" in leader_name:
            return ["Allow policy exceptions for experiments.", "Fast‑track path for innovation."]
        return ["Establish controls, measure, iterate."]

    def run_fmea(problem: str, decision: str, leader_name: str, want_eli5: bool):
        sev0, occ0, det0 = base_scores(problem, decision)
        sev, occ, det = style_adjusted_scores(sev0, occ0, det0, leader_name)
        rpn = sev * occ * det
        res = {
            "Failure Mode": f"Execution gaps and unintended consequences under {leader_name}.",
            "Effects": "Delays, cost overruns, quality issues, compliance risks, missed opportunities.",
            "Severity": sev,
            "Occurrence": occ,
            "Detection": det,
            "RPN": rpn,
            "Mitigation": mitigation_for_style(leader_name),
        }
        if want_eli5:
            res["ELI5"] = (
                f"ELI5: Severity is how big the problem is, Occurrence is how often it might happen, "
                f"Detection is how quickly we can spot it. Higher RPN needs attention. S={sev}, O={occ}, D={det}."
            )
        return res

    # Controls
    cc1, cc2, cc3 = st.columns([1.5, 1, 1])
    with cc1:
        chosen_case = st.selectbox("Pick a classic case (optional)", list(CASES.keys()))
        default_problem = CASES.get(chosen_case, "")
    with cc2:
        show_eli5 = st.checkbox("Show ELI5", value=True)
        compact = st.checkbox("Compact view", value=False)
    with cc3:
        delay = st.slider("Thinking delay per agent (s)", 0.0, 0.5, 0.10, 0.05)

    problem = st.text_area("Problem", value=default_problem, height=100, placeholder="Describe the business problem…")
    decision = st.text_area("Decision taken by CEO", height=100, placeholder="Describe the decision…")

    run_btn = st.button("Run FMEA with 10 Leadership Agents")

    if run_btn:
        if not problem.strip() or not decision.strip():
            st.warning("Please provide both Problem and Decision.")
        else:
            st.success("Running agents…")
            st.markdown("---")
            for leader, desc in LEADER_STYLES.items():
                placeholder = st.empty()
                placeholder.info(f"Thinking… {leader}")
                time.sleep(delay)
                placeholder.empty()
                box = st.container()
                with box:
                    st.subheader(leader)
                    st.caption(desc)
                    result = run_fmea(problem, decision, leader, want_eli5=show_eli5)
                    if compact:
                        st.markdown(f"**Failure Mode:** {result['Failure Mode']}")
                        st.markdown(f"**Effects:** {result['Effects']}")
                        st.markdown("**Mitigation:**")
                        for m in result["Mitigation"]:
                            st.markdown(f"- {m}")
                        if show_eli5 and "ELI5" in result:
                            st.write(result["ELI5"])
                    else:
                        st.markdown(f"**Failure Mode:** {result['Failure Mode']}")
                        st.markdown(f"**Effects:** {result['Effects']}")
                        st.markdown(f"**Scores:** S={result['Severity']}, O={result['Occurrence']}, D={result['Detection']} → RPN={result['RPN']}")
                        st.markdown("**Mitigation:**")
                        for m in result["Mitigation"]:
                            st.markdown(f"- {m}")
                        if show_eli5 and "ELI5" in result:
                            st.info(result["ELI5"])

# -----------------------------
# Tab 5 — 5W+1H Usefulness Analysis
# -----------------------------
with T5:
    st.subheader("5W + 1H — Usefulness Analysis")
    st.caption("Who uses this, What it does, When and Where to use, Why it helps, and How it works.")

    who = "CXOs, strategy teams, product leaders, ops, risk & compliance, and students."
    what = "Autonomous strategy exploration, KPI forecasting, risk scoring, execution visualization, and governance ledger in one place."
    when = "Quarterly planning, crisis simulations, board prep, portfolio reviews, and training."
    where = "Streamlit Cloud or local laptop; shareable links for reviews."
    why = "To make faster, explainable, multi-stakeholder decisions with transparent trade-offs."
    how = "Monte Carlo outcomes + persona voting + user priorities; FMEA for leadership risk; agentic progress; hashed ledger; exports."

    df_use = pd.DataFrame({
        "Question": ["Who", "What", "When", "Where", "Why", "How"],
        "Answer": [who, what, when, where, why, how],
    })

    st.table(df_use)

    st.download_button(
        "Download 5W1H CSV",
        df_use.to_csv(index=False).encode(),
        "usefulness_5w1h.csv",
        "text/csv",
    )

    txt = "\n".join([f"{q}: {a}" for q, a in zip(df_use["Question"], df_use["Answer"])])
    st.download_button("Download 5W1H TXT", txt, "usefulness_5w1h.txt", "text/plain")
