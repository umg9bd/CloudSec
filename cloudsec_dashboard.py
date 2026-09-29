"""
CloudSec-XAI Console -- a Streamlit dashboard over the REAL pipeline output.

Reads the risk scores the live pipeline writes (output/risk_scores.csv: per-event
HGT p_graph, LSTM p_sequence, ensemble risk_score, alert) and the per-principal
alert files (alerts/*.json), and renders the high-risk log stream, the structural
triplet, and a per-event breakdown.

Explainability -- the SHAP feature weights and the natural-language narrative --
is a PLACEHOLDER, clearly marked, to be wired to the XAI module later.

Run:
    streamlit run cloudsec_dashboard.py
Generate the numbers it reads first (once):
    python pipeline.py --files datasets/privilege-escalation/real_dataset_test.csv
"""

import glob
import json
import os

import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(HERE, "output", "risk_scores.csv")
ALERTS = os.path.join(HERE, "alerts")
RAW = os.path.join(HERE, "datasets", "privilege-escalation", "real_dataset_test.csv")
HIGH_RISK = 8.0   # the ">0.8 threshold" band in the log stream (risk_score is 0-10)

st.set_page_config(page_title="CloudSec-XAI Console", page_icon="🛡️", layout="wide")

CSS = """
<style>
  .stApp{background:#0a0b10;color:#e7e9f0;font-family:'JetBrains Mono',ui-monospace,monospace}
  #MainMenu,header,footer{visibility:hidden}
  .block-container{padding:1rem 1.4rem 2rem;max-width:100%}
  .up{text-transform:uppercase;letter-spacing:.09em;font-size:10.5px;color:#646b82;font-weight:600}
  .card{background:#14161f;border:1px solid #242835;border-radius:10px;overflow:hidden;margin-bottom:14px}
  .hd{display:flex;align-items:center;gap:10px;padding:10px 16px;border-bottom:1px solid #242835;
      background:linear-gradient(180deg,#171a24,#14161f);font-size:12.5px;font-weight:600}
  .live{margin-left:auto;width:8px;height:8px;border-radius:50%;background:#e0335f;box-shadow:0 0 8px #ff3d6e}
  .topbar{display:flex;align-items:center;gap:16px;padding:11px 16px;background:linear-gradient(180deg,#12131c,#0d0e15);
          border:1px solid #242835;border-radius:10px;margin-bottom:14px;font-size:12.5px;color:#9aa0b4}
  .topbar b{color:#e7e9f0} .dot{width:9px;height:9px;border-radius:50%;background:#3ecf8e;box-shadow:0 0 10px #3ecf8e}
  .alerts{margin-left:auto;background:#3a1226;color:#ff7ea3;border:1px solid #5a2038;border-radius:6px;
          padding:7px 12px;font-weight:700}
  .thead{display:grid;grid-template-columns:1.7fr 2fr .8fr .9fr 1.2fr;gap:12px;padding:9px 16px;
         border-bottom:1px solid #242835;background:#111320}
  .trow{display:grid;grid-template-columns:1.7fr 2fr .8fr .9fr 1.2fr;gap:12px;padding:11px 16px;align-items:center;
        border-bottom:1px solid #242835}
  .rk-crit{background:linear-gradient(90deg,rgba(224,51,95,.16),rgba(139,92,246,.05))}
  .rk-high{background:linear-gradient(90deg,rgba(139,92,246,.14),transparent)}
  .ev{color:#fff;font-weight:500} .who{color:#646b82;font-size:11px}
  .node{color:#c084fc} .rel{color:#646b82} .arrow{color:#8b5cf6}
  .risk{display:inline-block;font-weight:700;padding:3px 9px;border-radius:6px;color:#fff}
  .risk.crit{background:linear-gradient(90deg,#c1224a,#8b2fb0)} .risk.high{background:linear-gradient(90deg,#6d3fc0,#8b5cf6)}
  .flag{font-size:10.5px;color:#ffb3c9} .vel{font-size:11.5px;color:#9aa0b4}
  .ph{display:inline-flex;gap:6px;font-size:9.5px;letter-spacing:.08em;font-weight:700;color:#8b93ad;
      background:#1b1e2b;border:1px dashed #3a4059;border-radius:5px;padding:3px 8px;text-transform:uppercase}
  .json{background:#0b0c12;border:1px solid #242835;border-radius:8px;padding:14px;font-size:12px;
        line-height:1.7;white-space:pre;color:#9aa0b4;overflow-x:auto}
  .k{color:#c084fc} .s{color:#7ee0b0} .w{color:#ffb15c}
  .sfeat{margin-bottom:11px} .sfeat .r{display:flex;justify-content:space-between;font-size:11.5px;margin-bottom:4px}
  .sfeat .r .val{color:#c084fc;font-weight:600}
  .bar{height:8px;border-radius:4px;background:#20242f} .bar > i{display:block;height:100%;border-radius:4px;
       background:linear-gradient(90deg,#6d3fc0,#c084fc)}
  .cnode{display:inline-block;vertical-align:top;background:#191c27;border:1px solid #2f3444;border-radius:9px;
         padding:10px 13px;min-width:130px;margin:2px}
  .cnode .t{font-weight:600;color:#fff;font-size:12px} .cnode .s2{font-size:10.5px;color:#646b82}
  .cedge{display:inline-block;color:#8b5cf6;padding:0 8px;font-size:14px}
  .stat{font-size:26px;font-weight:700} .stat.crit{color:#e0335f} .stat.acc{color:#c084fc}
  .stat + .l{color:#646b82;font-size:10.5px;text-transform:uppercase;letter-spacing:.08em}
  .stSelectbox label{color:#9aa0b4 !important}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load():
    if not os.path.exists(SCORES):
        return None, None, None
    df = pd.read_csv(SCORES, low_memory=False)
    df = df.drop_duplicates("log_id", keep="last")
    raw = None
    if os.path.exists(RAW):
        r = pd.read_csv(RAW, low_memory=False, dtype=str)
        r["log_id"] = "real_dataset_test.csv:" + r.index.astype(str)
        raw = r.set_index("log_id")
    alerts = []
    for f in sorted(glob.glob(os.path.join(ALERTS, "*.json"))):
        try:
            alerts.append(json.load(open(f, encoding="utf-8")))
        except Exception:
            pass
    return df, raw, alerts


df, raw, alerts = load()
if df is None:
    st.error("No pipeline output found. Run:  python pipeline.py --files "
             "datasets/privilege-escalation/real_dataset_test.csv")
    st.stop()

n_alerts = len(alerts)
avg_risk = df["risk_score"].mean() / 10
high = df[df["risk_score"] >= HIGH_RISK].sort_values("risk_score", ascending=False).head(40)

st.markdown(
    f"""<div class="topbar"><span class="dot"></span>
    <span>User: <b>Admin</b></span><span>|</span>
    <span>System: <b style="color:#3ecf8e">Active</b></span><span>|</span>
    <span>Pipeline: <b>HGT + LSTM ensemble</b></span>
    <span class="alerts">⚠ {n_alerts} High-Risk Alert Principals</span></div>""",
    unsafe_allow_html=True)

# ── stat strip ──
c1, c2, c3, c4 = st.columns(4)
for col, val, lab, cls in [
    (c1, f"{len(df):,}", "Events scored", "acc"),
    (c2, f"{int((df['risk_score'] >= HIGH_RISK).sum()):,}", "High-risk (>0.8)", "crit"),
    (c3, f"{n_alerts}", "Alert principals", "crit"),
    (c4, f"{avg_risk:.2f}", "Avg risk score", "acc"),
]:
    col.markdown(f'<div class="stat {cls}">{val}</div><div class="l">{lab}</div>', unsafe_allow_html=True)

st.write("")

# ── high-risk log stream ──
def risk_cls(v):
    return "crit" if v >= 9.0 else "high"

rows_html = '<div class="thead up"><div>Log snippet</div><div>Structural triplet</div><div>Risk</div>' \
            '<div>Branch</div><div>Flags</div></div>'
for _, r in high.iterrows():
    rc = risk_cls(r["risk_score"])
    pg = "n/a" if pd.isna(r["p_graph"]) else f"{r['p_graph']:.2f}"
    branch = "LSTM only" if pd.isna(r["p_graph"]) else f"HGT {pg}"
    src = str(r["source_node"]).split("/")[-1][:18]
    tgt = str(r["target_node"]).split("/")[-1][:18]
    flag = "ALERT" if r["alert"] else "watch"
    rows_html += (
        f'<div class="trow rk-{rc}">'
        f'<div><div class="ev">{{{r["event_name"]}}}</div><div class="who">{str(r["username"])[-26:]}</div></div>'
        f'<div class="vel"><span class="node">[{src}]</span> <span class="rel">—{r["edge_type"]}<span class="arrow">▶</span></span> <span class="node">[{tgt}]</span></div>'
        f'<div><span class="risk {rc}">{r["risk_score"]:.2f}</span></div>'
        f'<div class="vel">{branch} · LSTM {r["p_sequence"]:.2f}</div>'
        f'<div class="flag">{flag}</div></div>')

st.markdown(f'<div class="card"><div class="hd">Real-time High Risk Logs (&gt;0.8 Threshold) '
            f'<span class="live"></span></div>{rows_html}</div>', unsafe_allow_html=True)

# ── selected-log breakdown ──
labels = [f"{r['risk_score']:.2f}  {r['event_name']}  ·  {str(r['username'])[-24:]}"
          for _, r in high.iterrows()]
pick = st.selectbox("Selected log for XAI breakdown", range(len(high)),
                    format_func=lambda i: labels[i]) if len(high) else None

if pick is not None:
    r = high.iloc[pick]
    lid = r["log_id"]
    # real CloudTrail-ish JSON from the raw event
    fields = {"eventName": r["event_name"], "eventSource": "", "userIdentity.arn": "",
              "sourceIPAddress": "", "userAgent": ""}
    if raw is not None and lid in raw.index:
        rr = raw.loc[lid]
        fields["eventSource"] = str(rr.get("event_source", ""))
        fields["userIdentity.arn"] = str(rr.get("principal_arn", ""))
        fields["sourceIPAddress"] = str(rr.get("source_ip", ""))
        fields["userAgent"] = str(rr.get("user_agent", ""))
    js = "{\n" + ",\n".join(
        f'  <span class="k">"{k.split(".")[-1]}"</span>: <span class="s">"{v}"</span>'
        for k, v in fields.items() if v and v != "nan") + "\n}"

    colA, colB = st.columns(2)
    with colA:
        st.markdown('<div class="up">Original CloudTrail (real event)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="json">{js}</div>', unsafe_allow_html=True)
        pgv = "n/a (LSTM only)" if pd.isna(r["p_graph"]) else f"{r['p_graph']:.3f}"
        st.markdown(
            f'<div style="margin-top:12px;font-size:12px;color:#9aa0b4">'
            f'HGT p_graph: <b style="color:#c084fc">{pgv}</b> &nbsp; · &nbsp; '
            f'LSTM p_sequence: <b style="color:#c084fc">{r["p_sequence"]:.3f}</b> &nbsp; · &nbsp; '
            f'ensemble risk: <b style="color:#ff7ea3">{r["risk_score"]:.2f}/10</b></div>',
            unsafe_allow_html=True)
    with colB:
        st.markdown('<div class="up">Feature Weights (SHAP) '
                    '<span class="ph">◇ Placeholder — XAI module not wired</span></div>',
                    unsafe_allow_html=True)
        # placeholder weights, deterministic per event type (clearly a mock)
        mock = [("Graph branch (HGT)", 40 if not pd.isna(r["p_graph"]) else 0),
                ("Sequence branch (LSTM)", int(r["p_sequence"] * 40)),
                ("Event velocity", 15), ("Privilege delta", 12), ("Rare role pair", 8)]
        mock = [(n, w) for n, w in mock if w > 0]
        mx = max(w for _, w in mock)
        bars = "".join(
            f'<div class="sfeat"><div class="r"><span>{n}</span><span class="val">+{w}%</span></div>'
            f'<div class="bar"><i style="width:{int(w/mx*100)}%"></i></div></div>' for n, w in mock)
        st.markdown(bars, unsafe_allow_html=True)

    # attack chain from the real triplet
    src = str(r["source_node"]).split("/")[-1]
    tgt = str(r["target_node"]).split("/")[-1]
    chain = (f'<span class="cnode"><span class="t">{str(r["username"])[-20:]}</span>'
             f'<span class="s2">acting principal</span></span>'
             f'<span class="cedge">──▶</span>'
             f'<span class="cnode"><span class="t">{r["event_name"]}</span>'
             f'<span class="s2">{r["edge_type"]}</span></span>'
             f'<span class="cedge">──▶</span>'
             f'<span class="cnode"><span class="t">{tgt}</span>'
             f'<span class="s2">target resource</span></span>')
    st.markdown(f'<div class="card"><div class="hd">Attack Chain (from structural triplet)</div>'
                f'<div style="padding:16px">{chain}</div></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="card"><div class="hd">Natural Language Explanation '
        '<span class="ph" style="margin-left:auto">◇ Placeholder — generated by XAI module</span></div>'
        '<div style="padding:16px;color:#646b82;font-size:12.5px">The natural-language narrative for this '
        'event will be generated by the explainability module and rendered here.</div></div>',
        unsafe_allow_html=True)

st.markdown(
    '<div style="color:#646b82;font-size:11px;margin-top:10px">Model: CloudSec-XAI · '
    'HGT + LSTM ensemble · numbers are the live pipeline output on the real held-out test capture · '
    'MITRE ATT&CK TA0004 / TA0008</div>', unsafe_allow_html=True)
