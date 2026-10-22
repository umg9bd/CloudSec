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
from prompt_injection_gate.pipeline_bridge import PipelineBridge
from prompt_injection_gate.schemas import ContextItem, Decision, ScreeningRequest, TRUST_LEVELS

st.set_page_config(page_title="Stage-1 Prompt Screening", page_icon="🛡️", layout="centered")

_COLOR = {Decision.SAFE: "#1a7f37", Decision.MALICIOUS: "#b42318", Decision.REVIEW_REQUIRED: "#b54708"}
_BG = {Decision.SAFE: "#e6f4ea", Decision.MALICIOUS: "#fbeae8", Decision.REVIEW_REQUIRED: "#fdf3e7"}
_ICON = {Decision.SAFE: "✅", Decision.MALICIOUS: "⛔", Decision.REVIEW_REQUIRED: "⚠️"}
_SEV_COLOR = {"critical": "#b42318", "high": "#b42318", "medium": "#b54708", "low": "#5b6572", "info": "#5b6572"}


def _sev(label):
    c = _SEV_COLOR.get(str(label).lower(), "#5b6572")
    return (f"<span style='background:{c};color:#fff;border-radius:4px;padding:1px 7px;"
            f"font-size:12px;font-weight:700'>{str(label).upper()}</span>")


@st.cache_resource(show_spinner="Loading screening model…")
def get_bridge():
    bridge = PipelineBridge(load_config())
    svc = bridge.service
    return bridge, svc.classifier.available, svc.classifier.version, svc.rules_version


def _risk_banner(s2):
    """A one-line risk read for the SAFE prompt, from Stage 2's security review."""
    gaps = s2.get("security_gaps", [])
    sev = {str(g.get("severity", "")).lower() for g in gaps}
    review = s2.get("requires_user_review")
    if {"critical", "high"} & sev:
        n = sum(1 for g in gaps if str(g.get("severity", "")).lower() in ("critical", "high"))
        st.error(f"⚠️ **Could be risky** — {n} high/critical security gap(s) must be addressed before this runs. "
                 "The changes below are already added to the prompt.")
    elif gaps:
        st.warning(f"**Some risk** — {len(gaps)} security gap(s) found; required changes are added below.")
    else:
        st.success("**Low risk** — no security gaps; the request already meets the applicable controls.")
    if review:
        st.caption("A human should review this before it is executed (`requires_user_review = true`).")


def _render_stage2(s2):
    status = s2.get("status")
    st.subheader(f"Stage 2 — security review · {status}")
    _risk_banner(s2)

    gaps = s2.get("security_gaps", [])
    added = s2.get("added_controls", [])
    satisfied = s2.get("satisfied_controls", [])
    deferred = s2.get("deferred_controls", [])
    questions = s2.get("clarification_questions", [])
    conflicts = s2.get("conflicts", [])
    c = st.columns(4)
    c[0].metric("Risks found", len(gaps))
    c[1].metric("Changes added", len(added))
    c[2].metric("Already met", len(satisfied))
    c[3].metric("Open questions", len(questions))

    if gaps:
        st.markdown("**Risks found**")
        for g in gaps:
            st.markdown(f"- {_sev(g.get('severity', 'info'))} &nbsp;`{g.get('control_id', '?')}` — "
                        f"{g.get('description', '')}", unsafe_allow_html=True)

    if added:
        st.markdown("**Required changes added to the prompt**")
        for a in added:
            with st.expander(f"{a.get('control_id', '?')} — {a.get('control_name', '')}  ·  {str(a.get('severity', '')).upper()}"):
                if a.get("clause"):
                    st.markdown("**Change:** " + a["clause"])
                if a.get("verification_expectation"):
                    st.markdown("**How to verify:** " + a["verification_expectation"])
                if a.get("reason"):
                    st.caption(a["reason"])

    if questions:
        st.markdown("**Open questions (answer before running)**")
        for q in questions:
            flag = " ⛔ blocking" if q.get("blocking") else ""
            st.markdown(f"- **{q.get('question', '')}**{flag}  \n  _{q.get('why_it_matters', '')}_")

    if satisfied:
        with st.expander(f"Already satisfied by the request ({len(satisfied)})"):
            for sdt in satisfied:
                st.markdown(f"- `{sdt.get('control_id', '?')}` — {sdt.get('verification_expectation', '')}")

    for label, items in (("Conflicts", conflicts), ("Deferred controls", deferred),
                         ("Warnings", s2.get("warnings", [])), ("Errors", s2.get("errors", []))):
        if items:
            st.markdown(f"**{label}:** " + ", ".join(
                (i.get("control_id") or i.get("message") or str(i)) if isinstance(i, dict) else str(i) for i in items))

    scope = s2.get("scope_findings", {}) or {}
    in_scope = scope.get("approved_services", [])
    out_scope = scope.get("out_of_scope_services", [])
    if in_scope or out_scope:
        st.caption("Services — in scope: " + (", ".join(in_scope) or "none")
                   + ((" · out of scope: " + ", ".join(out_scope)) if out_scope else ""))

    if s2.get("revised_prompt"):
        with st.expander("Hardened prompt (what Stage 3 receives)"):
            st.code(s2["revised_prompt"])


def _render_stage3(s3):
    contract = s3.get("stage3_contract", {}) or {}
    st.subheader(f"Stage 3 — predicted AWS activity · {contract.get('status')}")
    st.caption("What the agent's call would actually do, and the audit trail it would leave — "
               "the input the GNN/LSTM risk scorer (Stage 4) consumes.")

    actions = contract.get("proposed_actions", [])
    if actions:
        st.markdown("**Proposed AWS API calls (in order)**")
        st.dataframe({"step": [a.get("step") for a in actions],
                      "operation": [a.get("operation") for a in actions],
                      "service": [a.get("service") for a in actions],
                      "IAM permission": [a.get("iam_permission") for a in actions],
                      "why": [a.get("rationale", "") for a in actions]},
                     use_container_width=True, hide_index=True)

    seq = s3.get("downstream_event_sequence", [])
    if seq:
        st.markdown("**Predicted CloudTrail events:** " + " → ".join(f"`{e}`" for e in seq))
    rows = s3.get("downstream_gnn_structural_rows", [])
    if rows:
        with st.expander(f"Graph rows for the scorer ({len(rows)})"):
            st.dataframe({"edge_type": [r.get("edge_type") for r in rows],
                          "target_node": [r.get("target_node") for r in rows],
                          "state": [r.get("state") for r in rows]},
                         use_container_width=True, hide_index=True)

    cov = contract.get("prediction_coverage", {}) or {}
    if cov:
        ok = "complete within the catalogue scope" if cov.get("complete_within_scope") else "NOT complete within scope"
        st.caption(f"Coverage: {ok}.")
        for lim in cov.get("limitations", []):
            st.caption(f"• {lim}")
    for label, key in (("Assumptions", "assumptions"), ("Missing parameters", "missing_parameters"),
                       ("Unmodelled effects", "unmodelled_effects"), ("Open questions", "clarification_questions")):
        vals = contract.get(key, [])
        if vals:
            st.markdown(f"**{label}:** " + "; ".join(str(v) for v in vals))

    prov = contract.get("source_provenance", [])
    if prov:
        with st.expander(f"AWS documentation provenance ({len(prov)})"):
            for u in prov:
                st.markdown(f"- {u}")
    with st.expander("Full Stage 3 payload (JSON for Stage 4)"):
        st.json({"downstream_event_sequence": seq,
                 "downstream_gnn_structural_rows": rows,
                 "downstream_lstm_records": s3.get("downstream_lstm_records", {}),
                 "stage3_contract": contract})


def _render_downstream(bridge, resp):
    """Always run Stage 2 (security review + harden) and Stage 3 (predict events) for a SAFE prompt."""
    request = {
        "request_id": resp.request_id,
        "screening_status": bridge.cfg.handoff_screening_status,
        "original_prompt": resp.safe_prompt_for_next_stage,
    }
    with st.spinner("Stage 2: reviewing and hardening the prompt…"):
        s2 = bridge.run_stage2_inline(request)
    st.divider()
    if s2 is None:
        st.warning("Stage 2 (security_hardening) is not importable here — the file was still delivered for a running watcher to pick up.")
        return
    _render_stage2(s2)

    status = s2.get("status")
    if status not in ("HARDENED", "ALREADY_ADEQUATE") or not s2.get("revised_prompt"):
        st.info(f"Stage 3 does not run for a Stage 2 status of {status}.")
        return

    with st.spinner("Stage 3: predicting the AWS actions and CloudTrail events…"):
        s3 = bridge.run_stage3(resp.safe_prompt_for_next_stage, s2["revised_prompt"], resp.request_id)
    st.divider()
    if s3 is None:
        st.warning("Stage 3 (log-generation) is not importable here (is pydantic installed in this environment?).")
        return
    _render_stage3(s3)


def main():
    bridge, clf_ok, clf_ver, rules_ver = get_bridge()
    svc = bridge.service

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
        delivered_path = bridge.forward(resp)
        if delivered_path:
            st.caption(f"📤 Delivered to Stage 2 inbox: `{delivered_path}`")
        _render_downstream(bridge, resp)      # Stage 2 + Stage 3 always run for a SAFE prompt
    else:
        st.info("Nothing is forwarded. The prompt is held for the user or a reviewer — Stage 2 and Stage 3 do not run.")

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
