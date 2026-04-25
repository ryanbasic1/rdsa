"""Streamlit application entry point for TalentScout."""

from __future__ import annotations

import streamlit as st

from chatbot import TalentScoutChatbot
from utils import detect_sentiment, mask_profile_data

CONSENT_MESSAGE = (
    "Before we begin, please provide consent for temporary in-memory processing of your "
    "screening data. No data is stored permanently."
)

st.set_page_config(page_title="TalentScout AI Hiring Assistant", page_icon="TS", layout="wide")


def build_quick_actions(consent_given: bool, progress: dict, assessment_report: dict) -> list[str]:
    """Return contextual quick-action messages for faster candidate interaction."""
    if not consent_given:
        return []

    if assessment_report.get("active"):
        return ["skip", "next", "Can you give me a hint?"]

    if progress.get("finished"):
        return [
            "start assessment",
            "Give me one more React question",
            "What should I improve for this role?",
            "exit",
        ]

    return [
        "Why do you need this detail?",
        "Can you continue to the next step?",
        "exit",
    ]


def get_input_placeholder(consent_given: bool, progress: dict, assessment_report: dict) -> str:
    """Return dynamic placeholder text for chat input based on current state."""
    if not consent_given:
        return "Consent is required before input can be submitted."

    if assessment_report.get("active"):
        return "Answer the active technical question, or type skip."

    if progress.get("finished"):
        return "Ask for deeper practice questions or type start assessment."

    next_field = (progress.get("next_field") or "next step").replace("_", " ")
    return f"Share your {next_field}..."

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;700&family=Manrope:wght@400;500;700&display=swap');

        :root {
            --bg-1: #f6f8fd;
            --bg-2: #edf3ff;
            --bg-3: #fff4e9;
            --card: #ffffff;
            --text: #13283f;
            --muted: #4a5f77;
            --accent: #1456a0;
            --accent-soft: #dcebff;
            --accent-2: #0f766e;
            --outline: #d6e3f2;
            --user-bubble: #1b365d;
            --user-text: #f3f8ff;
        }

        .stApp {
            font-family: 'Manrope', sans-serif;
            background:
                radial-gradient(1200px 540px at -6% -10%, #d7e9ff 0%, transparent 62%),
                radial-gradient(760px 360px at 106% 6%, #ffe9ce 0%, transparent 62%),
                radial-gradient(500px 220px at 74% 88%, #d7f2ee 0%, transparent 64%),
                linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 44%, #ffffff 100%);
            color: var(--text);
        }

        .main .block-container {
            max-width: 1200px;
            padding-top: 1.2rem;
            padding-bottom: 5rem;
        }

        .ts-header {
            background: linear-gradient(145deg, #ffffff 0%, #eef5ff 65%, #f4fbff 100%);
            border: 1px solid var(--outline);
            border-radius: 20px;
            padding: 1.15rem 1.3rem;
            box-shadow: 0 20px 34px rgba(16, 58, 103, 0.11);
            margin-bottom: 0.95rem;
            animation: fadeUp 380ms ease-out;
        }

        .ts-title {
            margin: 0;
            font-size: 1.85rem;
            font-weight: 700;
            font-family: 'Sora', sans-serif;
            color: var(--text);
            letter-spacing: 0.2px;
        }

        .ts-subtitle {
            margin-top: 0.35rem;
            color: var(--muted);
            font-size: 0.97rem;
        }

        .ts-note {
            margin-top: 0.7rem;
            border: 1px dashed #c6dbf4;
            background: #f8fcff;
            color: #2f4f71;
            border-radius: 12px;
            padding: 0.62rem 0.75rem;
            font-size: 0.84rem;
        }

        .ts-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 0.55rem;
            margin-top: 0.9rem;
        }

        .ts-stat {
            border: 1px solid #d7e8fb;
            border-radius: 12px;
            padding: 0.55rem 0.7rem;
            background: #f9fcff;
        }

        .ts-stat-label {
            color: #3d5875;
            font-size: 0.74rem;
            margin-bottom: 0.15rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-weight: 700;
        }

        .ts-stat-value {
            color: #173b66;
            font-size: 1.02rem;
            font-weight: 700;
        }

        .ts-pill {
            display: inline-block;
            margin-bottom: 0.45rem;
            padding: 0.24rem 0.58rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        @keyframes fadeUp {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0px);
            }
        }

        .sentiment-chip {
            display: inline-block;
            padding: 0.24rem 0.6rem;
            border-radius: 999px;
            background: #e3f5ef;
            color: #13695e;
            font-size: 0.76rem;
            font-weight: 600;
            margin-bottom: 0.6rem;
        }

        .ts-side-card {
            position: sticky;
            top: 0.8rem;
            border: 1px solid var(--outline);
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border-radius: 14px;
            padding: 0.85rem;
            box-shadow: 0 10px 22px rgba(22, 59, 106, 0.08);
            margin-bottom: 0.75rem;
        }

        .ts-side-title {
            margin: 0;
            font-size: 0.95rem;
            color: #173b66;
            font-weight: 700;
        }

        .ts-helper {
            margin-top: 0.28rem;
            color: #48617e;
            font-size: 0.83rem;
            line-height: 1.4;
        }

        [data-testid="stChatMessage"] {
            border: 1px solid var(--outline);
            border-radius: 15px;
            background: var(--card);
            box-shadow: 0 10px 22px rgba(14, 48, 90, 0.07);
            padding: 0.25rem 0.45rem;
            margin-bottom: 0.62rem;
            animation: fadeUp 320ms ease-out;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span {
            color: #1a2f4c !important;
            line-height: 1.5;
        }

        [data-testid="stChatMessageAvatarUser"] {
            background: #ffd9cf;
            color: #7d1f0f;
        }

        [data-testid="stChatMessageAvatarAssistant"] {
            background: #d9f4ee;
            color: #0f5c52;
        }

        [data-testid="stChatInput"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.55), #ffffff 65%);
            border-top: 1px solid var(--outline);
            padding-top: 0.5rem;
        }

        [data-testid="stChatInput"] textarea {
            font-family: 'Manrope', sans-serif !important;
        }

        [data-testid="stButton"] button {
            border-radius: 10px;
            border: 1px solid #bdd5ef;
            background: linear-gradient(180deg, #f8fbff 0%, #eef5ff 100%);
            color: #153d69;
            font-weight: 600;
        }

        [data-testid="stButton"] button:hover {
            border-color: #9ec0e6;
            color: #0e335e;
            transform: translateY(-1px);
        }

        .ts-quick-title {
            margin: 0.15rem 0 0.5rem 0;
            color: #315476;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }

        @media (max-width: 768px) {
            .ts-title {
                font-size: 1.5rem;
            }

            .main .block-container {
                padding-top: 1rem;
                padding-bottom: 4.4rem;
            }

            .ts-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

main_col, side_col = st.columns([1.6, 0.9], gap="large")

if "bot" not in st.session_state:
    st.session_state.bot = TalentScoutChatbot()

if "consent_given" not in st.session_state:
    st.session_state.consent_given = False

if "clear_pending" not in st.session_state:
    st.session_state.clear_pending = False

if "llm_status" not in st.session_state:
    st.session_state.llm_status = st.session_state.bot.llm.check_connection()

if "messages" not in st.session_state:
    if st.session_state.consent_given:
        opening_message = st.session_state.bot.start_conversation()
        st.session_state.messages = [{"role": "assistant", "content": opening_message}]
    else:
        st.session_state.messages = [{"role": "assistant", "content": CONSENT_MESSAGE}]

if "sentiment" not in st.session_state:
    st.session_state.sentiment = "neutral"

llm_ok, llm_message = st.session_state.llm_status

progress = st.session_state.bot.get_progress()
profile = st.session_state.bot.get_profile_data()
assessment_report = st.session_state.bot.get_assessment_report()
collected_tech_count = len(profile.get("tech_stack") or [])
input_placeholder = get_input_placeholder(st.session_state.consent_given, progress, assessment_report)

with main_col:
    action_col_1, action_col_2 = st.columns([1, 1], gap="small")
    with action_col_1:
        if st.button("Start New Screening", help="Clear chat and restart the screening flow."):
            st.session_state.bot = TalentScoutChatbot()
            st.session_state.consent_given = False
            st.session_state.messages = [{"role": "assistant", "content": CONSENT_MESSAGE}]
            st.session_state.sentiment = "neutral"
            st.session_state.llm_status = st.session_state.bot.llm.check_connection()
            st.session_state.clear_pending = False
            st.rerun()

    with action_col_2:
        if st.button("Re-check Groq API", help="Run a live connectivity check with your current key."):
            st.session_state.llm_status = st.session_state.bot.llm.check_connection()
            st.rerun()

    if not st.session_state.consent_given:
        st.warning(
            "Consent required: Please confirm consent to continue screening. Data is processed in-memory for this session only."
        )
        if st.button("I Consent and Start Screening"):
            st.session_state.consent_given = True
            st.session_state.bot = TalentScoutChatbot()
            st.session_state.messages = [{"role": "assistant", "content": st.session_state.bot.start_conversation()}]
            st.session_state.sentiment = "neutral"
            st.session_state.clear_pending = False
            st.rerun()

    st.markdown(
        f"""
        <div class="ts-header">
            <div class="ts-pill">AI Hiring Assistant</div>
            <h1 class="ts-title">TalentScout</h1>
            <div class="ts-subtitle">Smart pre-screening with context-aware technical questioning</div>
            <div class="ts-note">Tip: Use quick actions for faster navigation. You can always ask "why" at any step.</div>
            <div class="ts-grid">
                <div class="ts-stat">
                    <div class="ts-stat-label">Progress</div>
                    <div class="ts-stat-value">{progress['percent']}%</div>
                </div>
                <div class="ts-stat">
                    <div class="ts-stat-label">Fields Completed</div>
                    <div class="ts-stat-value">{progress['completed']} / {progress['total']}</div>
                </div>
                <div class="ts-stat">
                    <div class="ts-stat-label">Technologies</div>
                    <div class="ts-stat-value">{collected_tech_count}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if llm_ok:
        st.success(f"Groq API status: {llm_message}")
    else:
        st.error(f"Groq API status: {llm_message}")
        st.caption("If API status is failing, question generation will use fallback questions.")

    if st.session_state.sentiment != "neutral":
        st.markdown(
            f"<span class='sentiment-chip'>Detected tone: {st.session_state.sentiment}</span>",
            unsafe_allow_html=True,
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    quick_actions = build_quick_actions(st.session_state.consent_given, progress, assessment_report)
    pending_input = None
    if quick_actions:
        st.markdown("<div class='ts-quick-title'>Quick Actions</div>", unsafe_allow_html=True)
        quick_cols = st.columns(len(quick_actions), gap="small")
        for idx, label in enumerate(quick_actions):
            with quick_cols[idx]:
                if st.button(label, key=f"quick_action_{idx}_{label[:12]}"):
                    pending_input = label

    user_input = st.chat_input(
        input_placeholder,
        disabled=not st.session_state.consent_given,
    )

    if user_input:
        pending_input = user_input

    if pending_input:
        st.session_state.sentiment = detect_sentiment(pending_input)
        st.session_state.messages.append({"role": "user", "content": pending_input})

        with st.chat_message("user"):
            st.markdown(pending_input)

        with st.chat_message("assistant"):
            with st.spinner("TalentScout is processing your response..."):
                response = st.session_state.bot.handle_message(pending_input)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

with side_col:
    next_field = progress.get("next_field")
    friendly_next = (next_field or "Screening complete").replace("_", " ").title()
    st.markdown(
        f"""
        <div class="ts-side-card">
            <h3 class="ts-side-title">Session Guide</h3>
            <div class="ts-helper">Next focus: <b>{friendly_next}</b></div>
            <div class="ts-helper">Use <b>exit</b>, <b>quit</b>, or <b>bye</b> to close the chat at any time.</div>
            <div class="ts-helper">For a deeper round, type <b>start assessment</b> after questions are generated.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(progress["percent"] / 100)
    st.caption(f"Progress: {progress['completed']} of {progress['total']} fields collected")
    st.caption(f"Assessment answers captured: {assessment_report['answered']} / {assessment_report['queued']}")

    if assessment_report.get("summary"):
        with st.expander("Recruiter Summary", expanded=True):
            st.markdown(str(assessment_report["summary"]))

    has_collected_data = any(
        value for key, value in profile.items() if key != "tech_stack"
    ) or bool(profile.get("tech_stack"))

    if has_collected_data:
        with st.expander("Candidate Snapshot (In-Memory)", expanded=progress["finished"]):
            st.write(mask_profile_data(profile))
            st.caption("PII is masked in this preview. Data remains in memory for this session and is not persisted.")

    if st.button("Clear Session Data"):
        st.session_state.clear_pending = True

    if st.session_state.clear_pending:
        st.warning("Confirm clear action: this will remove chat, profile, and assessment data from session memory.")
        confirm_col, cancel_col = st.columns(2)
        with confirm_col:
            if st.button("Confirm Clear"):
                st.session_state.bot = TalentScoutChatbot()
                st.session_state.messages = [{"role": "assistant", "content": CONSENT_MESSAGE}]
                st.session_state.sentiment = "neutral"
                st.session_state.consent_given = False
                st.session_state.clear_pending = False
                st.session_state.llm_status = st.session_state.bot.llm.check_connection()
                st.rerun()
        with cancel_col:
            if st.button("Cancel Clear"):
                st.session_state.clear_pending = False
                st.rerun()
