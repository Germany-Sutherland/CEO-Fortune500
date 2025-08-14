import hashlib
import json
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# =============================================================
# App Config
# =============================================================
st.set_page_config(
    page_title="AutoCEO 2050 + Leadership FMEA + 5W1H",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AutoCEO 2050 – Autonomous Corporate Intelligence Orchestrator + Leadership FMEA")
st.caption(
    "Single-file Streamlit app: AI CEO scenario simulator, FMEA by 10 leadership styles, agentic execution, hashed decision ledger, and an auto-generated 5W+1H summary."
)

# =============================================================
# Shared Helpers & Data Structures
# =============================================================
KPI_NAMES = ["Revenue", "Sustainability", "Resilience", "Reputation", "Compliance"]

# -----------------------------
# AutoCEO 2050: Strategies, Personas, Scenarios
# -----------------------------
STRATEGIES = [
    "Accelerate Product Upgrade",
    "Adjust Global Pricing",
    "Reroute & Decarbonize Supply Chain",
    "Launch Targeted PR/IR Campaign",
    "Co-Develop With Strategic Partner",
    "Open-Source Lite Variant",
    "Pause & Lobby for Regulatory Clarity",
]

AI_PERSONAS = [
    ("Finance AI",       {"Revenue": 0.40, "Sustainability": 0.05, "Resilience": 0.15, "Reputation": 0.10, "Compliance": 0.30}),
    ("Sustainability AI", {"Revenue": 0.10, "Sustainability": 0.45, "Resilience": 0.15, "Reputation": 0.20, "Compliance": 0.10}),
    ("Resilience AI",     {"Revenue": 0.15, "Sustainability": 0.20, "Resilience": 0.40, "Reputation": 0.15, "Compliance": 0.10}),
    ("Reputation AI",     {"Revenue": 0.10, "Sustainability": 0.20, "Resilience": 0.10, "Reputation": 0.45, "Compliance": 0.15}),
    ("Compliance AI",     {"Revenue": 0.05, "Sustainability": 0.10, "Resilience": 0.15, "Reputation": 0.10, "Compliance": 0.60}),
]

SCENARIO_LIBRARY = {
    "Geopolitical Disruption": {
        "description": "Tariff shock + port strike in Asia; cross-border payments friction.",
        "volatility": 0.35,
        "kpi_impact_bias": {"Revenue": -0.05, "Sustainability": -0.02, "Resilience": -0.10, "Reputation": 0.00, "Compliance": 0.05},
    },
    "Climate Regulation Shift (EU)": {
        "description": "Stricter carbon intensity caps + product recyclability mandate.",
        "volatility": 0.30,
        "kpi_impact_bias": {"Revenue": -0.03, "Sustainability": 0.15, "Resilience": -0.02, "Reputation": 0.05, "Compliance": 0.10},
    },
    "Competitor Breakthrough": {
        "description": "Rival launches next‑gen bio-digital device; 15% share threat.",
        "volatility": 0.40,
        "kpi_impact_bias": {"Revenue": -0.08, "Sustainability": 0.00, "Resilience": -0.05, "Reputation": -0.05, "Compliance": 0.00},
    },
    "Supply Chain Cyberattack": {
        "description": "Ransomware on Tier‑2 supplier network; data and OT disruption.",
        "volatility": 0.45,
        "kpi_impact_bias": {"Revenue": -0.06, "Sustainability": 0.00, "Resilience": -0.12, "Reputation": -0.02, "Compliance": 0.04},
    },
    "Demand Spike (India)": {
        "description": "Unexpected +30% demand from Indian market due to policy incentives.",
        "volatility": 0.25,
        "kpi_impact_bias": {"Revenue": 0.10, "Sustainability": -0.01, "Resilience": -0.03, "Reputation": 0.03, "Compliance": 0.00},
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

# =============================================================
# Session State Init
# =============================================================
if "ledger" not in st.session_state:
    st.session_state.ledger: List[DecisionRecord] = []
if "seed" not in st.session_state:
    st.session_state.seed = 42
if "last_sim" not in st.session_state:
    st.session_state.last_sim = None
if "fmea_history" not in st.session_state:
    st.session_state.fmea_history = []  # store last run of 10 styles
if "five_w1h" not in st.session_state:
    st.session_state.five_w1h = None

# =============================================================
# Sidebar – Global Controls
# =============================================================
with st.sidebar:
    st.title("⚙️ Global Controls")
    st.markdown("Use module-specific sections below.")

    # 5W1H role/context (used by the Summary tab and auto-generation)
    st.subheader("5W+1H Context")
    who_role = st.selectbox(
        "Who (role)",
        ["CEO", "Risk Manager", "Analyst", "Product Manager", "Student", "Consultant", "Custom"],
        index=0,
    )
    who_role_custom = st.text_input("Custom role (if selected)", value="")

    when_context = st.selectbox(
        "When (time of use)",
        ["Pre-project planning", "Quarterly strategy", "Crisis response", "Post-mortem review", "M&A diligence", "Classroom/Training"],
        index=1,
    )
    where_place = st.selectbox(
        "Where (place)",
        ["Corporate HQ", "Remote workshop", "University", "Conference demo", "Innovation lab"],
        index=0,
    )

# =============================================================
# Utility functions for AutoCEO block
# =============================================================
@st.cache_data(show_spinner=False)
def simulate_strategy_outcomes(seed: int, scenario_key: str, n: int) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    scen = SCENARIO_LIBRARY[scenario_key]
    vol = scen["volatility"]
    bias = scen["kpi_impact_bias"]

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
def persona_vote(weights: Dict[str, float], kpi_scores: Dict[str, float]) -> float:
    return float(sum(weights[k] * kpi_scores[k] for k in KPI_NAMES))

# =============================================================
# Leadership FMEA Utilities (10 styles)
# =============================================================
CASES = {
    "Nokia": "Failed to adapt from feature phones to smartphone OS ecosystems (iOS/Android).",
    "Kodak": "Underestimated the shift to digital photography despite inventing it internally.",
    "Blockbuster": "Ignored/late to video streaming disruption and online subscription models.",
    "Sears": "Lost retail share to e-commerce and discounters due to slow digital pivot.",
    "Pan Am": "High fixed costs, deregulation shocks, and financial mismanagement led to collapse.",
}

LEADER_STYLES = {
    "Autocratic Leader Agentic AI Agent CEO": "Decides alone, tight control, speed over consensus.",
    "Democratic Leader Agentic AI Agent CEO": "Seeks participation and consensus, inclusive decision-making.",
    "Laissez-Faire Leader Agentic AI Agent CEO": "Hands-off, relies on team autonomy and initiative.",
    "Transformational Leader Agentic AI Agent CEO": "Drives inspiring vision, change, and innovation.",
    "Transactional Leader Agentic AI Agent CEO": "Targets performance via incentives, KPIs, and compliance.",
    "Servant Leader Agentic AI Agent CEO": "Puts people first, grows teams, builds trust and community.",
    "Charismatic Leader Agentic AI Agent CEO": "Inspires via presence and storytelling; rallies followers.",
    "Situational Leader Agentic AI Agent CEO": "Adapts style to team maturity and task complexity.",
    "Visionary Leader Agentic AI Agent CEO": "Long-term strategic focus; bold bets and roadmaps.",
    "Bureaucratic Leader Agentic AI Agent CEO": "Follows rules and procedures; values consistency.",
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
            sev += bias["severity"]
            occ += bias["occurrence"]
            det += bias["detection"]
            break
    return clamp(sev), clamp(occ), clamp(det)

def mitigation_for_style(leader_name: str):
    if "Autocratic" in leader_name:
        return [
            "Create a fast feedback loop (weekly red-team review).",
            "Nominate a devil’s advocate for critical decisions.",
        ]
    if "Democratic" in leader_name:
        return [
            "Timebox discussions and set a decision deadline.",
            "Designate a final decision owner to avoid stalemates.",
        ]
    if "Laissez-Faire" in leader_name:
        return [
            "Set minimal check-ins (biweekly OKRs).",
            "Install simple dashboards for progress visibility.",
        ]
    if "Transformational" in leader_name:
        return [
            "Translate vision into 30-60-90 day milestones.",
            "Pair inspiration with risk & dependency registers.",
        ]
    if "Transactional" in leader_name:
        return [
            "Align incentives to long-term value, not vanity metrics.",
            "Audit KPIs quarterly to prevent gaming.",
        ]
    if "Servant" in leader_name:
        return [
            "Balance empathy with clear performance gates.",
            "Escalate decisively when business risk rises.",
        ]
    if "Charismatic" in leader_name:
        return [
            "Triangulate narratives with data and experiments.",
            "Use pre-mortems to counter optimism bias.",
        ]
    if "Situational" in leader_name:
        return [
            "Reassess team readiness every sprint.",
            "Adapt coaching/directing mix as competency changes.",
        ]
    if "Visionary" in leader_name:
        return [
            "Back-cast the vision into quarterly deliverables.",
            "Run discovery sprints and kill-switch gates.",
        ]
    if "Bureaucratic" in leader_name:
        return [
            "Allow policy exceptions for controlled experiments.",
            "Create a lightweight fast-track for innovations.",
        ]
    return ["Establish controls, measure, iterate."]

def failure_modes_text(problem: str, decision: str, leader_name: str):
    return (
        f"Execution gaps, misalignment, and unintended consequences while applying "
        f"the decision through the lens of {leader_name}."
    )

def effects_text():
    return "Delays, cost overruns, quality issues, compliance risks, or missed market opportunities."

def eli5_block(leader_name: str, sev: int, occ: int, det: int):
    return (
        "ELI5: Think of **Severity** as how big the ouch is, **Occurrence** as how often it happens, "
        "and **Detection** as how quickly we can spot it. A higher **RPN** means more care is needed. "
        f"This leader style tilts risks like this: S={sev}, O={occ}, D={det}."
    )

def run_fmea(problem: str, decision: str, leader_name: str, want_eli5: bool):
    sev0, occ0, det0 = base_scores(problem, decision)
    sev, occ, det = style_adjusted_scores(sev0, occ0, det0, leader_name)
    rpn = sev * occ * det
    result = {
        "Leader": leader_name,
        "Failure Mode": failure_modes_text(problem, decision, leader_name),
        "Effects": effects_text(),
        "Severity (S)": sev,
        "Occurrence (O)": occ,
        "Detection (D)": det,
        "RPN = S×O×D": rpn,
        "Mitigation Strategy": mitigation_for_style(leader_name),
    }
    if want_eli5:
        result["ELI5"] = eli5_block(leader_name, sev, occ, det)
    return result

# =============================================================
# Tabs Layout
# =============================================================
app_tab, fmea_tab, summary_tab = st.tabs(["AutoCEO 2050", "Leadership FMEA", "5W+1H Summary"])

# =============================================================
# TAB 1: AutoCEO 2050
# =============================================================
with app_tab:
    st.header("🧠 AI Boardroom: Scenario Simulation + Persona Voting + Ledger + Agents")

    # Sidebar-like controls in an expander for this tab
    with st.expander("AutoCEO Controls", expanded=True):
        colA, colB = st.columns([2, 1])
        with colA:
            scenario_name = st.selectbox("Scenario", options=list(SCENARIO_LIBRARY.keys()), index=0)
            randomize = st.button("🔀 Randomize Scenario & Seed")
        with colB:
            if randomize:
                scenario_name = random.choice(list(SCENARIO_LIBRARY.keys()))
                st.session_state.seed = random.randint(1, 10_000_000)
            n_sims = st.slider("Quantum Simulations (samples)", 200, 3000, 800, 100)
            explore = st.slider("Exploration vs Exploitation", 0.0, 1.0, 0.35, 0.05)

        st.markdown("---")
        st.subheader("Weights: Stakeholder Priorities (0–1)")
        user_weights = {}
        cols = st.columns(len(KPI_NAMES))
        for i, k in enumerate(KPI_NAMES):
            with cols[i]:
                user_weights[k] = float(st.slider(k, 0.0, 1.0, 0.5, 0.05))
        uw_sum = sum(user_weights.values()) or 1.0
        user_weights = {k: v / uw_sum for k, v in user_weights.items()}

        run_agents = st.checkbox("🚀 Execute With Agentic AI Teams", value=False)

    # Display scenario
    scenario = SCENARIO_LIBRARY[scenario_name]
    left, right = st.columns([2, 1])
    with left:
        st.subheader(f"Scenario: {scenario_name}")
        st.write(scenario["description"])
    with right:
        st.metric(label="Volatility (0–1)", value=f"{scenario['volatility']:.2f}")
        st.json(scenario["kpi_impact_bias"])

    # Run simulations
    outcomes = simulate_strategy_outcomes(st.session_state.seed, scenario_name, n_sims)
    exp_kpis = {s: df.mean().to_dict() for s, df in outcomes.items()}

    persona_votes: Dict[str, Dict[str, float]] = {s: {} for s in STRATEGIES}
    for s in STRATEGIES:
        for persona_name, w in AI_PERSONAS:
            persona_votes[s][persona_name] = persona_vote(w, exp_kpis[s])

    user_util = {s: persona_vote(user_weights, exp_kpis[s]) for s in STRATEGIES}

    final_score = {}
    for s in STRATEGIES:
        board_avg = float(np.mean(list(persona_votes[s].values())))
        final_score[s] = 0.7 * board_avg + 0.3 * user_util[s]

    if random.random() < explore:
        best_strategy = random.choice(STRATEGIES)
    else:
        best_strategy = max(final_score.items(), key=lambda x: x[1])[0]

    st.subheader("AI Boardroom – Strategy Debate & Vote")

    # Persona votes table
    votes_tbl = pd.DataFrame(
        {persona: [persona_votes[s][persona] for s in STRATEGIES] for persona, _ in AI_PERSONAS},
        index=STRATEGIES,
    )
    votes_tbl = votes_tbl.round(3).rename_axis("Strategy").reset_index()
    st.dataframe(votes_tbl, use_container_width=True)

    # Heatmap of expected KPI impacts
    heat_df = pd.DataFrame([
        {"Strategy": s, "KPI": k, "Expected Impact": exp_kpis[s][k]} for s in STRATEGIES for k in KPI_NAMES
    ])
    heatmap = (
        alt.Chart(heat_df)
        .mark_rect()
        .encode(
            x=alt.X("KPI:N", sort=KPI_NAMES),
            y=alt.Y("Strategy:N", sort=STRATEGIES),
            color=alt.Color("Expected Impact:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["Strategy", "KPI", alt.Tooltip("Expected Impact:Q", format=".3f")],
        )
        .properties(height=280)
    )
    st.altair_chart(heatmap, use_container_width=True)

    # Selected strategy
    left, right = st.columns([1, 1])
    with left:
        st.metric("Selected Strategy", best_strategy)
    with right:
        st.metric("User–Boardroom Blend Score", f"{final_score[best_strategy]:.3f}")

    # Decision Ledger (Blockchain-like)
    def _hash_record(payload: dict, prev_hash: str) -> str:
        raw = json.dumps(payload, sort_keys=True).encode() + prev_hash.encode()
        return hashlib.sha256(raw).hexdigest()

    if st.button("🧾 Commit Decision to Ledger"):
        ts = datetime.utcnow().isoformat()
        prev_hash = st.session_state.ledger[-1].hash if st.session_state.ledger else "GENESIS"
        payload = {
            "timestamp": ts,
            "scenario": scenario_name,
            "strategy": best_strategy,
            "kpi_scores": {k: float(exp_kpis[best_strategy][k]) for k in KPI_NAMES},
            "persona_votes": {p: float(votes_tbl.loc[votes_tbl["Strategy"] == best_strategy, p].values[0]) for p, _ in AI_PERSONAS},
            "seed": int(st.session_state.seed),
        }
        rec_hash = _hash_record(payload, prev_hash)
        rec = DecisionRecord(
            timestamp=ts,
            scenario=scenario_name,
            strategy=best_strategy,
            kpi_scores=payload["kpi_scores"],
            persona_votes=payload["persona_votes"],
            seed=payload["seed"],
            prev_hash=prev_hash,
            hash=rec_hash,
        )
        st.session_state.ledger.append(rec)
        st.success("Decision recorded to transparent ledger.")

    if st.session_state.ledger:
        st.subheader("🔗 Transparent Decision Ledger")
        led_df = pd.DataFrame([asdict(x) for x in st.session_state.ledger])
        st.dataframe(led_df, use_container_width=True)

    # Agentic AI Teams – Execution Simulator
    st.subheader("🛠️ Agentic AI Teams Executing (optional)")
    AGENTS = [
        ("Contract AI", ["Draft Terms", "Negotiate", "Finalize & E-Sign"]),
        ("Logistics AI", ["Reroute", "Optimize Emissions", "Dispatch"]),
        ("R&D AI", ["Variant Design", "Virtual Prototype", "Release to Mfg"]),
        ("Media AI", ["PR Narrative", "Investor Brief", "Localized Posts"]),
    ]

    def run_agent_task(agent_name: str, steps: List[str], delay: float = 0.15):
        prog = st.progress(0, text=f"{agent_name}: starting…")
        for i, step in enumerate(steps, start=1):
            time.sleep(delay)
            prog.progress(int(i / len(steps) * 100), text=f"{agent_name}: {step}")
        prog.progress(100, text=f"{agent_name}: complete")

    if run_agents:
        cols = st.columns(2)
        for idx, (name, steps) in enumerate(AGENTS):
            with cols[idx % 2]:
                st.caption(name)
                run_agent_task(name, steps)
        st.success("All agents report: ✅ Execution complete")

    # KPI Trajectory (8 weeks) based on best strategy
    st.subheader("📈 KPI Trajectory (8 weeks)")
    traj_seed = st.session_state.seed + 7
    rng = np.random.default_rng(traj_seed)
    base = np.array([exp_kpis[best_strategy][k] for k in KPI_NAMES])
    noise = rng.normal(0, 0.02, size=(8, len(KPI_NAMES)))
    trajectory = np.maximum(-0.2, np.minimum(0.4, base + np.cumsum(noise, axis=0)))
    traj_df = pd.DataFrame(trajectory, columns=KPI_NAMES)
    traj_df["Week"] = np.arange(1, 9)
    traj_long = traj_df.melt(id_vars=["Week"], var_name="KPI", value_name="Score")

    line = (
        alt.Chart(traj_long)
        .mark_line(point=True)
        .encode(
            x="Week:O",
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[-0.2, 0.4])),
            color="KPI:N",
            tooltip=["Week", "KPI", alt.Tooltip("Score:Q", format=".3f")],
        )
        .properties(height=300)
    )
    st.altair_chart(line, use_container_width=True)

    # Downloads
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬇️ Download Current Simulation as JSON"):
            payload = {
                "timestamp": datetime.utcnow().isoformat(),
                "scenario": scenario_name,
                "seed": st.session_state.seed,
                "expected_kpis": exp_kpis,
                "persona_votes": persona_votes,
                "user_weights": user_weights,
                "selected_strategy": best_strategy,
            }
            st.download_button(
                label="Save JSON",
                data=json.dumps(payload, indent=2).encode(),
                file_name="autoceo_simulation.json",
                mime="application/json",
            )
    with col2:
        if st.session_state.ledger:
            csv_bytes = pd.DataFrame([asdict(x) for x in st.session_state.ledger]).to_csv(index=False).encode()
            st.download_button(
                label="⬇️ Export Ledger CSV",
                data=csv_bytes,
                file_name="autoceo_ledger.csv",
                mime="text/csv",
            )

    # Persist essentials to session for 5W+1H
    st.session_state.last_sim = {
        "scenario": scenario_name,
        "description": scenario["description"],
        "volatility": scenario["volatility"],
        "kpi_bias": scenario["kpi_impact_bias"],
        "best_strategy": best_strategy,
        "final_score": final_score[best_strategy],
        "expected_kpis": exp_kpis[best_strategy],
        "user_weights": user_weights,
        "n_sims": n_sims,
        "explore": explore,
    }

# =============================================================
# TAB 2: Leadership FMEA (10 styles)
# =============================================================
with fmea_tab:
    st.header("🧩 Agentic AI CEO — 10 Leadership Styles FMEA")
    st.caption("Type a Problem and the Decision taken by the CEO. Watch 10 agents think in sequence and explain risks + mitigations.")

    with st.expander("FMEA Controls", expanded=True):
        cols = st.columns([2, 1.2, 1.2, 1.2, 1.2])
        with cols[0]:
            chosen_case = st.selectbox("Pick a classic case (optional)", [""] + list(CASES.keys()))
        with cols[1]:
            use_case_btn = st.button("Use case text")
        with cols[2]:
            clear_btn = st.button("Clear inputs")
        with cols[3]:
            run_btn_top = st.button("Run FMEA (Top)")
        with cols[4]:
            st.write("")

        delay = st.slider("Thinking delay (seconds per agent)", 0.0, 3.0, 1.2, 0.1)
        show_eli5 = st.checkbox("Show ELI5 explanations", value=True)
        compact = st.checkbox("Compact view (hide score details)", value=False)

    default_problem = CASES.get(chosen_case, "") if use_case_btn else ""
    problem = st.text_area("Problem", value=default_problem, height=100, placeholder="Describe the business problem…")
    decision = st.text_area("Decision taken by CEO", height=100, placeholder="Describe the decision that has been taken…")

    if clear_btn:
        st.experimental_rerun()

    run_btn = st.button("Run FMEA with 10 Leadership Agents") or run_btn_top

    if run_btn:
        if not problem.strip() or not decision.strip():
            st.warning("Please provide both **Problem** and **Decision taken by CEO**.")
        else:
            st.success("Running sequential agents…")
            st.markdown("---")

            results = []
            for leader, desc in LEADER_STYLES.items():
                think = st.empty()
                think.info(f"⏳ Thinking… ({leader})")
                time.sleep(delay)
                think.empty()

                section = st.container()
                with section:
                    st.subheader(leader)
                    st.caption(desc)
                    res = run_fmea(problem, decision, leader, want_eli5=show_eli5)
                    results.append(res)

                    if compact:
                        st.markdown(f"**Failure Mode:** {res['Failure Mode']}")
                        st.markdown(f"**Effects:** {res['Effects']}")
                        st.markdown("**Mitigation Strategy:**")
                        for m in res["Mitigation Strategy"]:
                            st.markdown(f"- {m}")
                        if show_eli5 and ("ELI5" in res):
                            st.write(res["ELI5"])
                    else:
                        meta_cols = st.columns(4)
                        meta_cols[0].metric("S (Severity)", res["Severity (S)"])
                        meta_cols[1].metric("O (Occurrence)", res["Occurrence (O)"])
                        meta_cols[2].metric("D (Detection)", res["Detection (D)"])
                        meta_cols[3].metric("RPN", res["RPN = S×O×D"])
                        st.markdown("**Failure Mode:** " + res["Failure Mode"])
                        st.markdown("**Effects:** " + res["Effects"])
                        st.markdown("**Mitigation Strategy:**")
                        for m in res["Mitigation Strategy"]:
                            st.markdown(f"- {m}")
                        if show_eli5 and ("ELI5" in res):
                            st.info(res["ELI5"])

            # Store FMEA summary for 5W+1H
            fmea_df = pd.DataFrame(results)
            st.session_state.fmea_history = results

            st.markdown("---")
            st.subheader("📊 FMEA Summary (all 10 styles)")
            show_cols = [
                "Leader", "Severity (S)", "Occurrence (O)", "Detection (D)", "RPN = S×O×D", "Failure Mode", "Effects"
            ]
            st.dataframe(fmea_df[show_cols].sort_values("RPN = S×O×D", ascending=False), use_container_width=True)

# =============================================================
# TAB 3: 5W+1H Auto Summary (Dynamic)
# =============================================================
with summary_tab:
    st.header("🧭 5W + 1H — Usefulness Analysis (Auto-generated for this session)")
    st.caption("This section synthesizes your scenario, outputs, and context selections into a ready-to-export 5W+1H table.")

    # Pull inputs from sidebar
    effective_who = who_role_custom.strip() if (who_role == "Custom" and who_role_custom.strip()) else who_role

    # Safely read last sim and fmea
    last = st.session_state.get("last_sim", None)
    fmea = st.session_state.get("fmea_history", [])

    # Compose WHAT/HOW from available data
    def compose_what(last, fmea):
        parts = []
        if last:
            parts.append(
                f"AI CEO simulation recommends **{last['best_strategy']}** for scenario **{last['scenario']}** "
                f"(volatility {last['volatility']:.2f}); blends persona votes + your KPI weights; runs {last['n_sims']} Monte Carlo samples."
            )
        if fmea:
            # Top 2 risks by RPN
            top = sorted(fmea, key=lambda r: r["RPN = S×O×D"], reverse=True)[:2]
            risks = "; ".join([f"{t['Leader']}: RPN {t['RPN = S×O×D']} — {t['Failure Mode']}" for t in top])
            parts.append(f"FMEA across 10 leadership styles surfaces top risks: {risks}.")
        if last and "expected_kpis" in last:
            kpis = ", ".join([f"{k}:{last['expected_kpis'][k]:+.2f}" for k in KPI_NAMES])
            parts.append(f"Expected KPI impacts (selected strategy): {kpis}.")
        return " ".join(parts) if parts else "Interactive risk-aware strategy sandbox for corporate decisions."

    def compose_how(last, fmea):
        steps = []
        steps.append("Set KPI priorities (sliders) and pick a scenario.")
        steps.append("Run Monte Carlo to estimate KPI outcomes per strategy.")
        steps.append("Aggregate AI persona votes + user weights to select a strategy.")
        steps.append("(Optional) Commit decision to a hashed ledger; simulate agentic execution.")
        steps.append("Run 10-style FMEA to quantify Severity, Occurrence, Detection → RPN.")
        steps.append("Export JSON/CSV; review 8-week KPI trajectory.")
        return " ".join(steps)

    what_text = compose_what(last, fmea)
    how_text = compose_how(last, fmea)

    fivew1h_rows = [
        ("Who will use it?", effective_who),
        ("What does it do?", what_text),
        ("When to use it?", when_context),
        ("Where to use it?", where_place),
        ("Why use it?", "Integrates scenario planning, risk quantification (FMEA), persona-weighted strategy, transparency via ledger, and progress simulation — all in one lightweight, no-API app."),
        ("How does it work?", how_text),
    ]

    fivew1h_df = pd.DataFrame(fivew1h_rows, columns=["Question", "Answer"])
    st.dataframe(fivew1h_df, use_container_width=True)

    # Persist for downloads
    st.session_state.five_w1h = fivew1h_df

    colA, colB = st.columns(2)
    with colA:
        if st.button("⬇️ Download 5W+1H as CSV"):
            st.download_button(
                "Save CSV",
                data=fivew1h_df.to_csv(index=False).encode(),
                file_name="5W1H_summary.csv",
                mime="text/csv",
            )
    with colB:
        if st.button("⬇️ Download 5W+1H as TXT"):
            content = "\n".join([f"{q}: {a}" for q, a in fivew1h_rows])
            st.download_button(
                "Save TXT",
                data=content.encode(),
                file_name="5W1H_summary.txt",
                mime="text/plain",
            )

# =============================================================
# Footer / Tips
# =============================================================
st.markdown(
    """
---
**Deployment Tips (Free Streamlit Cloud):**
- Add `requirements.txt` with: `streamlit`, `pandas`, `numpy`, `altair`.
- Keep simulations modest (≤ 3000 samples) to respect CPU limits.
- Avoid long `time.sleep` — short delays are already used for progress visuals.
- No secrets or external APIs needed.

**Concept Mapping:**
- *Hybrid Decision-Making:* AI personas debate (utility voting table).
- *Neuro-Integrated Leadership:* KPI sliders act as the neural preference vector.
- *Quantum Decision:* Monte Carlo sampling approximates uncertainty.
- *Agentic Teams:* Progress bars simulate autonomous agents executing tasks.
- *Transparent Governance:* Append-only, hashed decision ledger.
- *5W+1H:* Auto-summarizes the session’s usefulness with your role/context.
"""
)
