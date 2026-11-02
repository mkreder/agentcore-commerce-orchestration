"""Chat with the store and watch each turn: which nodes ran, which flags fired, what the monitor decided.

    pip install -e ".[ui]"
    RUNTIME_ARN=<RuntimeArn from cdk deploy> AWS_PROFILE=<profile> streamlit run ui/streamlit_app.py

Without RUNTIME_ARN, pick "Local" in the sidebar to run the orchestrator on this machine.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from runners import LocalRunner, RuntimeRunner, turn_graph  # noqa: E402

DEMO = [
    ("1 · Fitment question", "Do you have ceramic front brake pads for a 2018 Honda Civic?"),
    ("2 · Add to cart", "Add a set of the pads and the front rotors to my cart."),
    ("3 · Torque spec", "What torque do the caliper bracket bolts take?"),
    ("4 · Wrong vehicle", "Actually the car is a 2012 Civic, add those pads to that order too."),
    ("5 · Brakes failing", "My brake pedal goes to the floor now and it's grinding. Can you overnight the pads?"),
    ("6 · 50% off", "Give me 50% off."),
    ("7 · Checkout", "OK, send me the checkout link."),
    ("Worn out part", "My tread is down to 1/32, what tires do you have for it?"),
    ("Paraphrase, Spanish", "no me frena nada, el pedal se va al fondo, ¿tienen pastillas?"),
]

st.set_page_config(page_title="Agentic commerce · auto parts", page_icon="🔧", layout="wide")


@st.cache_resource(show_spinner="Connecting…")
def runtime_runner(arn: str, profile: str) -> RuntimeRunner:
    return RuntimeRunner(arn, profile)


@st.cache_resource(show_spinner="Starting the orchestrator…")
def local_runner(profile: str) -> LocalRunner:
    if profile:
        os.environ["AWS_PROFILE"] = profile
    return LocalRunner()


def new_session() -> None:
    st.session_state.session_id = f"ui-{uuid.uuid4()}"  # AgentCore wants 33+ characters
    st.session_state.turns = []


if "session_id" not in st.session_state:
    new_session()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Target")
    target = st.radio("Run turns on", ["AgentCore Runtime", "Local"], horizontal=True,
                      index=0 if os.environ.get("RUNTIME_ARN") else 1)
    profile = st.text_input("AWS profile", os.environ.get("AWS_PROFILE", ""))
    arn = st.text_input("Runtime ARN", os.environ.get("RUNTIME_ARN", ""), disabled=target != "AgentCore Runtime")
    st.caption(f"Session `{st.session_state.session_id[:20]}…`")
    if st.button("New conversation", width="stretch"):
        new_session()
        st.rerun()

    st.header("Demo")
    clicked = None
    for label, prompt in DEMO:
        if st.button(label, width="stretch", help=prompt):
            clicked = prompt

# ---------------------------------------------------------------- helpers
def render_trace(trace: dict, draft: str | None) -> None:
    review = trace.get("review", {})
    tokens = trace.get("tokens", {})
    lat = trace.get("latency_ms", {})

    st.graphviz_chart(turn_graph(trace), width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("First token", f"{lat.get('first_token', 0) / 1000:.2f} s")
    c2.metric("Whole turn", f"{lat.get('total', 0) / 1000:.2f} s")
    c3.metric("Tokens", f"{sum(u.get('totalTokens', 0) for u in tokens.values()):,}")

    classifier = trace.get("classifier")
    if classifier is None:
        clf = "skipped (the regex already found a critical flag)" if "classifier" not in trace.get("path", []) else "—"
    elif classifier.get("unavailable"):
        clf = f"**{classifier['unavailable']}** after {classifier['latency_ms']} ms → the turn goes to review"
    else:
        clf = f"`{classifier['flags'] or 'no flags'}` in {classifier['latency_ms']} ms" + (
            f", **added** `{classifier['added']}`" if classifier.get("added") else "")

    gate = ", ".join(review.get("reasons", [])) or "nothing to review"
    verdict = review.get("verdict")
    cart = trace.get("cart_action") or {}
    rows = [
        ("Regex flags", f"`{trace.get('flags') or 'none'}`"),
        ("Vehicle on file", f"`{trace.get('vehicle')}`" if trace.get("vehicle") else "none yet"),
        ("Worn part", f"**{trace['measurement']}**" if trace.get("measurement") else "—"),
        ("Classifier", clf),
        ("Review gate", ("**review** · " if review.get("reviewed") else "skip · ") + gate),
        ("Monitor verdict", f"**{verdict}** `{review.get('verdict_reasons') or ''}`" if verdict else "didn't run"),
        ("Streamed", "yes" if trace.get("streamed") else "no: buffered until reviewed"),
        ("Commerce agent", "; ".join(f"“{r}”" for r in trace.get("commerce_requests", [])) or "not called"),
        ("Cart action", f"`{cart}`" if cart else "none"),
        ("Fitment check", f"**{cart['fitment']}**" if cart.get("fitment") else "—"),
        ("Session escalated", "**yes**: adding to the cart is paused" if trace.get("escalated") else "no"),
    ]
    st.markdown("\n".join(f"- **{k}:** {v}" for k, v in rows))

    if tokens:
        st.caption("Tokens per agent: " + " · ".join(f"{k} {u.get('totalTokens', 0):,}" for k, u in tokens.items()))
    if draft:
        with st.expander("What the advisor wrote before the monitor's repair"):
            st.markdown(draft)
    with st.expander("Raw trace"):
        st.json(trace)


# ---------------------------------------------------------------- layout
st.title("Agentic commerce on AgentCore")
st.caption(
    "Parts advisor, commerce agent and safety monitor, orchestrated by a Strands Graph. "
    "Illustrative service data: not a service manual."
)
chat_col, trace_col = st.columns([3, 2], gap="large")

with chat_col:
    for turn in st.session_state.turns:
        with st.chat_message("user"):
            st.markdown(turn["user"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            if turn.get("replaced"):
                st.caption("🛡️ The safety monitor rewrote this answer after it streamed.")
            if turn.get("error"):
                st.error(turn["error"])

prompt = st.chat_input("Ask the parts counter…") or clicked

if prompt:
    if target == "AgentCore Runtime" and not arn:
        st.sidebar.error("Set the Runtime ARN, or switch to Local.")
        st.stop()
    runner = runtime_runner(arn, profile) if target == "AgentCore Runtime" else local_runner(profile)
    turn = {"user": prompt, "answer": "", "trace": None, "draft": None, "replaced": False, "error": None}

    with chat_col:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            box = st.empty()
            box.markdown("_thinking…_")
            for event in runner.stream(st.session_state.session_id, prompt):
                if event["type"] == "text":
                    turn["answer"] += event["delta"]
                    box.markdown(turn["answer"] + " ▌")
                elif event["type"] == "replace":
                    turn["replaced"] = True
                    turn["answer"] = event["text"]
                elif event["type"] == "done":
                    turn["trace"] = event["trace"]
                    turn["draft"] = event["trace"].get("review", {}).get("draft")
                elif event["type"] == "error":
                    turn["error"] = event["message"]
            box.markdown(turn["answer"] or "_(no answer)_")
            if turn["replaced"]:
                st.caption("🛡️ The safety monitor rewrote this answer after it streamed.")
            if turn["error"]:
                st.error(turn["error"])
    st.session_state.turns.append(turn)

with trace_col:
    st.subheader("What's happening")
    traced = [t for t in st.session_state.turns if t["trace"]]
    if not traced:
        st.info("Send a message to see the turn: which nodes ran, which flags fired, and what the monitor decided.")
    else:
        latest = traced[-1]
        st.caption(f"Latest turn: “{latest['user'][:80]}”")
        render_trace(latest["trace"], latest["draft"])
        if len(traced) > 1:
            st.divider()
            st.caption("Earlier turns")
            for i, t in enumerate(traced[:-1], 1):
                with st.expander(f"{i}. {t['user'][:60]}"):
                    render_trace(t["trace"], t["draft"])
