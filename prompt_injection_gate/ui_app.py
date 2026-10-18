"""
ui_app.py
=========
A small Streamlit UI for the Stage-1 gate: paste a prompt (optionally with
conversation history and untrusted context), screen it, and see the decision.

It calls the SAME ScreeningService the pipeline would, so what you see here is
exactly what Stage 2 would receive. It changes nothing in the existing pipeline.

Run (module venv has streamlit + transformers):
    prompt_injection_gate/.venv/Scripts/python.exe -m streamlit run prompt_injection_gate/ui_app.py

If no trained checkpoint is present the classifier is unavailable and the gate
fails closed (every prompt -> REVIEW_REQUIRED); train one with
training/train_eval_transformer.py first.
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from prompt_injection_gate.config import load_config
from prompt_injection_gate.schemas import ContextItem, Decision, ScreeningRequest, TRUST_LEVELS
from prompt_injection_gate.screening_service import ScreeningService

st.set_page_config(page_title="Stage-1 Prompt Screening", page_icon="🛡️", layout="centered")

_COLOR = {Decision.SAFE: "#1a7f37", Decision.MALICIOUS: "#b42318", Decision.REVIEW_REQUIRED: "#b54708"}
_BG = {Decision.SAFE: "#e6f4ea", Decision.MALICIOUS: "#fbeae8", Decision.REVIEW_REQUIRED: "#fdf3e7"}
_ICON = {Decision.SAFE: "✅", Decision.MALICIOUS: "⛔", Decision.REVIEW_REQUIRED: "⚠️"}


@st.cache_resource(show_spinner="Loading screening model…")
def get_service():
    svc = ScreeningService(load_config())
    return svc, svc.classifier.available, svc.classifier.version, svc.rules_version


def main():
    svc, clf_ok, clf_ver, rules_ver = get_service()

    st.title("🛡️ Stage-1 Prompt Screening")
    st.caption("Screens a prompt before the cloud-security workflow. A prompt is forwarded "
               "to the next stage **only** when the decision is SAFE.")

    cols = st.columns(2)
    cols[0].metric("Rules catalogue", rules_ver)
    cols[1].metric("Classifier", "available" if clf_ok else "UNAVAILABLE")
    if not clf_ok:
        st.warning("No trained classifier loaded — the gate is failing closed, so every prompt "
                   "returns REVIEW_REQUIRED. Train one with training/train_eval_transformer.py.")

    import uuid
    mode = st.radio(
        "Mode", ["Single prompt", "Conversation (multi-turn memory)"], horizontal=True,
        help="Single prompt: each screen is independent and repeatable. "
             "Conversation: the gate remembers this session's turns to catch a gradual (crescendo) jailbreak.")
    conversation = mode.startswith("Conversation")

    sid = None
    if conversation:
        if "sid" not in st.session_state:
            st.session_state.sid = "ui-" + uuid.uuid4().hex[:12]
        sid = st.session_state.sid
        remembered = svc.sessions.history(sid)
        bar = st.columns([3, 1])
        bar[0].caption(f"Conversation `{sid}` — {len(remembered)} turn(s) remembered")
        if bar[1].button("New conversation", use_container_width=True):
            svc.forget_session(sid)
            st.session_state.sid = "ui-" + uuid.uuid4().hex[:12]
            st.rerun()
        if remembered:
            with st.expander("Turns remembered so far (fed as context)"):
                for i, t in enumerate(remembered, 1):
                    st.text(f"{i}. {t}")

    with st.form("screen"):
        label = "Prompt (this turn)" if conversation else "Prompt"
        prompt = st.text_area(label, height=120,
                              placeholder="e.g. Create an S3 bucket for the public website assets.")
        with st.expander("Optional: untrusted context (document / tool output the agent would read)"):
            ctx_text = st.text_area("Context content", height=80, label_visibility="collapsed")
            ctx_trust = st.selectbox("Context trust level", [t for t in TRUST_LEVELS if t != "user_prompt"])
        submitted = st.form_submit_button("Screen prompt", type="primary", use_container_width=True)

    if not submitted:
        return
    if not prompt.strip():
        st.error("Enter a prompt to screen.")
        return

    context = [ContextItem(ctx_text, source="ui", trust=ctx_trust)] if ctx_text.strip() else []
    # Single-prompt mode sends no session_id, so screening the same prompt is repeatable.
    req = ScreeningRequest(prompt=prompt, untrusted_context=context, session_id=sid)
    resp = svc.screen(req)
    d = resp.decision

    st.markdown(
        f"<div style='padding:16px;border-radius:10px;background:{_BG[d]};"
        f"border:1px solid {_COLOR[d]}33'>"
        f"<span style='font-size:20px;font-weight:700;color:{_COLOR[d]}'>{_ICON[d]} {d.value}</span>"
        f"<br><span style='color:#333'>{resp.explanation}</span></div>",
        unsafe_allow_html=True)

    c = st.columns(3)
    c[0].metric("Forwarded to Stage 2?", "Yes" if resp.handoff_allowed else "No")
    c[1].metric("Threat categories", len(resp.threat_categories))
    c[2].metric("Latency", f"{resp.latency_ms:.0f} ms")

    if resp.handoff_allowed:
        st.success("safe_prompt_for_next_stage (unaltered original):")
        st.code(resp.safe_prompt_for_next_stage or "")
    else:
        st.info("Nothing is forwarded. The prompt is held for the user or a reviewer.")

    if resp.threat_categories:
        st.write("**Threat categories:** " + ", ".join(resp.threat_categories))

    with st.expander("Why — rules, scores and reason codes"):
        st.write("**Triggered rules:** " + (", ".join(resp.triggered_rule_ids) or "none"))
        st.write("**Reason codes:** " + ", ".join(resp.reason_codes))
        if resp.classifier_scores:
            st.write("**Classifier scores** (not calibrated probabilities):")
            scores = sorted(resp.classifier_scores.items(), key=lambda kv: -kv[1])
            st.dataframe({"signal": [k for k, _ in scores],
                          "score": [round(v, 3) for _, v in scores]},
                         use_container_width=True, hide_index=True)
        if resp.error_status:
            st.write(f"**Classifier status:** {resp.error_status}")
        st.caption(f"rules {resp.rules_version} · classifier {clf_ver} · request {resp.request_id}")

    with st.expander("Downstream contract (JSON)"):
        st.json(resp.to_contract())


if __name__ == "__main__":
    main()
