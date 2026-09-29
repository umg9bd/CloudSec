"""
CloudSec-XAI Console -- a Streamlit dashboard over the live pipeline's output.

Reads what pipeline.py writes as it scores:
  output/risk_scores.csv   every scored event, in arrival order: HGT p_graph, LSTM p_sequence,
                           ensemble risk_score, alert, fast_lane (+ the rule that fired)
  alerts/*.json            one per principal per input file, with the ensemble explanations
                           (ensemble_explain.py) of its top events and of every fast-lane event

and shows:
  - fast-lane alerts (defense-evasion actions flagged by rule), in their own panel;
  - the processed-log stream, every event in arrival order, new ones appended at the bottom,
    scrollable and filterable (all / alerts / fast-lane);
  - "why was this flagged": click an event to see its ensemble explanation -- how much of the
    risk came from each model, and what drove each model's score.

Run (with the live pipeline, one command -- the page refreshes as new scores land):
    run.cmd  /  ./run.sh        ->  http://localhost:8501
Or on its own, over output a pipeline run already wrote:
    streamlit run cloudsec_dashboard.py
"""

import glob
import html
import json
import os
import time

import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(HERE, "output", "risk_scores.csv")
ALERTS = os.path.join(HERE, "alerts")
CONFIG = os.path.join(HERE, "pipeline_config.json")
REFRESH_SECONDS = 5
TABLE_HEIGHT = 440
LIVE_WITHIN_SECONDS = 30          # scores written this recently -> the pipeline counts as live
MIN_RELATED_SHARE = 0.05          # as ensemble_explain.MIN_RELATED_SHARE: smaller graph neighbours aren't shown
MIN_DRIVER_SHARE = 0.25           # as ensemble_explain.MIN_DRIVER_SHARE: a model below this isn't a reason
FILTERS = {"All events": None, "Alerts": "alert", "Fast-lane": "fast_lane"}

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
  .hd .sub{margin-left:auto;font-weight:400;color:#646b82;font-size:11px}
  .live{width:8px;height:8px;border-radius:50%;background:#e0335f;box-shadow:0 0 8px #ff3d6e}
  .topbar{display:flex;align-items:center;gap:16px;padding:11px 16px;background:linear-gradient(180deg,#12131c,#0d0e15);
          border:1px solid #242835;border-radius:10px;margin-bottom:14px;font-size:12.5px;color:#9aa0b4}
  .topbar b{color:#e7e9f0} .dot{width:9px;height:9px;border-radius:50%;background:#3ecf8e;box-shadow:0 0 10px #3ecf8e}
  .dot.idle{background:#646b82;box-shadow:none}
  .alerts{margin-left:auto;background:#3a1226;color:#ff7ea3;border:1px solid #5a2038;border-radius:6px;
          padding:7px 12px;font-weight:700}
  .stat{font-size:26px;font-weight:700} .stat.crit{color:#e0335f} .stat.acc{color:#c084fc} .stat.fl{color:#ff9f43}
  .stat + .l{color:#646b82;font-size:10.5px;text-transform:uppercase;letter-spacing:.08em}

  /* fast-lane panel: rule-based alerts, deliberately louder than model alerts */
  .flpanel{border:1px solid #7a3a12;background:#1c120b;border-radius:10px;margin-bottom:14px;overflow:hidden}
  .flpanel .hd{background:linear-gradient(90deg,#4a1d08,#1c120b);border-bottom:1px solid #7a3a12;color:#ffb877}
  .fllist{max-height:210px;overflow-y:auto}
  .flrow{display:grid;grid-template-columns:160px 1fr 1.2fr 2fr 90px;gap:12px;padding:9px 16px;
         border-bottom:1px solid #3a2112;font-size:12px;align-items:center}
  .flrow .act{color:#fff;font-weight:600} .flrow .why{color:#ffb877} .flrow .t{color:#9aa0b4}
  .flbadge{display:inline-block;background:#ff9f43;color:#1c120b;font-weight:800;font-size:10px;
           letter-spacing:.08em;padding:2px 7px;border-radius:4px}
  .empty{padding:12px 16px;color:#646b82;font-size:12px}

  /* explanation */
  .banner{border-radius:8px;padding:10px 14px;margin-bottom:12px;font-size:12.5px}
  .banner.fl{background:#2a1608;border:1px solid #7a3a12;color:#ffb877}
  .banner.al{background:#2a0f1c;border:1px solid #5a2038;color:#ff9ab8}
  .banner.ok{background:#101a16;border:1px solid #1f3a2e;color:#7ee0b0}
  .summary{background:#0b0c12;border:1px solid #242835;border-radius:8px;padding:12px 14px;font-size:12.5px;
           line-height:1.7;color:#cfd3e0;margin-bottom:12px}
  .share{display:flex;height:26px;border-radius:6px;overflow:hidden;margin:6px 0 4px;font-size:11.5px;font-weight:700}
  .share > div{display:flex;align-items:center;padding:0 10px;white-space:nowrap;color:#fff}
  .share .g{background:linear-gradient(90deg,#6d3fc0,#8b5cf6)} .share .s{background:linear-gradient(90deg,#1f6f8b,#2fa3c7)}
  .kv{font-size:11.5px;color:#9aa0b4;margin-bottom:10px} .kv b{color:#e7e9f0}
  .note{font-size:11px;color:#646b82;margin:4px 0 10px}
  /* technical detail: label | bar | value, red right = raised the risk, green left = lowered it */
  .legend{display:flex;gap:18px;align-items:center;font-size:11.5px;color:#9aa0b4;margin:2px 0 14px}
  .legend .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
  .sw.pos{background:#ff5c7c} .sw.neg{background:#3ecf8e} .sw.mag{background:#8b7cf6}
  .th{font-size:12.5px;font-weight:600;color:#e7e9f0;margin-bottom:6px}
  .th small{display:block;font-weight:400;font-size:10.5px;color:#646b82;margin-top:1px}
  .dv{display:grid;grid-template-columns:minmax(0,1fr) 84px 64px;gap:10px;align-items:center;font-size:12px;
      padding:6px 0;border-bottom:1px solid #1b1e29}
  .dv .lab{color:#cfd3e0;line-height:1.35;overflow-wrap:anywhere}   /* wraps: the whole label stays readable */
  .dv .when{display:block;color:#646b82;font-size:10.5px}
  .dv .track{position:relative;height:9px;background:#1b1e29;border-radius:3px}
  .dv .track:not(.one):after{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:#4a5068}
  .dv .track i{position:absolute;top:0;bottom:0;border-radius:2px}
  .dv i.pos{background:#ff5c7c} .dv i.neg{background:#3ecf8e} .dv i.mag{background:#8b7cf6}
  .dv .v{text-align:right;font-weight:600;font-variant-numeric:tabular-nums} .dv .v.pos{color:#ff8fa8}
  .dv .v.neg{color:#7ee0b0} .dv .v.mag{color:#b7abff}
  .cnode{display:inline-block;vertical-align:top;background:#191c27;border:1px solid #2f3444;border-radius:9px;
         padding:10px 13px;min-width:130px;margin:2px}
  .cnode .t{font-weight:600;color:#fff;font-size:12px;display:block} .cnode .s2{font-size:10.5px;color:#646b82}
  .cedge{display:inline-block;color:#8b5cf6;padding:0 8px;font-size:14px}
  .stSelectbox label, .stRadio label{color:#9aa0b4 !important}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
esc = lambda v: html.escape(str(v))


# ── data ─────────────────────────────────────────────────────────────────
@st.cache_data
def alert_threshold():
    try:
        with open(CONFIG, encoding="utf-8") as f:
            return float(json.load(f)["alert_threshold"]) * 10
    except Exception:
        return 5.0


@st.cache_data(max_entries=4)
def load_scores(mtime):
    """Every scored event in arrival order. Positions must stay stable while rows are appended
    (the table's selection is a row position), so duplicates keep their FIRST occurrence."""
    df = pd.read_csv(SCORES, low_memory=False).drop_duplicates("log_id", keep="first").reset_index(drop=True)
    for col, default in (("fast_lane", False), ("fast_lane_reason", None), ("principal_arn", None),
                         ("source_ip", None)):     # files from before these columns existed
        if col not in df:
            df[col] = default
    df["alert"] = df["alert"].astype(str).str.lower().eq("true")
    df["fast_lane"] = df["fast_lane"].astype(str).str.lower().eq("true")
    return df


@st.cache_data(max_entries=5000)
def read_alert(path, mtime):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:          # a file the pipeline is still writing: picked up on the next refresh
        return None


def load_alerts():
    alerts = []
    for f in glob.glob(os.path.join(ALERTS, "alert_*.json")):
        a = read_alert(f, os.path.getmtime(f))
        if a:
            alerts.append(a)
    explanations = {e["log_id"]: e for a in alerts for e in a.get("explanations", [])}
    by_group = {(a["principal"], a["source_file"]): a for a in alerts}
    return alerts, explanations, by_group


# ── rendering helpers ────────────────────────────────────────────────────
def short(node):
    return str(node).rstrip("/").split("/")[-1].split(":")[-1][:40]


def flag_of(r):
    return "⚡ FAST-LANE" if r["fast_lane"] else "▲ ALERT" if r["alert"] else ""


def render_fast_lane(fl):
    rows = ""
    for _, r in fl.iloc[::-1].iterrows():       # newest first
        rows += (f'<div class="flrow"><span class="t">{esc(str(r["timestamp"])[:19])}</span>'
                 f'<span>{esc(r["username"])}</span><span class="act">{esc(r["event_name"])}</span>'
                 f'<span class="why">{esc(r["fast_lane_reason"] or "critical action")}</span>'
                 f'<span>risk {r["risk_score"]:.2f}</span></div>')
    body = f'<div class="fllist">{rows}</div>' if rows else '<div class="empty">No fast-lane events yet.</div>'
    st.markdown(f'<div class="flpanel"><div class="hd"><span class="flbadge">FAST-LANE</span> '
                f'Rule-based alerts: defense evasion, raised the moment the event arrives, whatever the models '
                f'score<span class="sub">{len(fl)} event(s) · newest first · pick "Fast-lane" below to see why'
                f'</span></div>{body}</div>', unsafe_allow_html=True)


def unseen(name):
    """The LSTM maps actions outside its training vocabulary to <UNK>."""
    return "an action unseen in training" if name in ("<UNK>", "<PAD>", None) else name


def ago(minutes):
    return f"{minutes * 60:.0f} s" if minutes < 1 else f"{minutes:.0f} min"


def render_event_header(r, threshold):
    if r["fast_lane"]:
        st.markdown(f'<div class="banner fl"><span class="flbadge">FAST-LANE</span> &nbsp;<b>'
                    f'{esc(r["fast_lane_reason"] or "critical action")}</b> -- flagged by rule the moment it '
                    f'arrived, whatever the models score.</div>', unsafe_allow_html=True)
    elif r["alert"]:
        st.markdown(f'<div class="banner al">▲ <b>ALERT</b> -- risk {r["risk_score"]:.2f} out of 10 '
                    f'(alerts start at {threshold:.2f}).</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="banner ok">Not flagged -- risk {r["risk_score"]:.2f} out of 10 '
                    f'(alerts start at {threshold:.2f}).</div>', unsafe_allow_html=True)
    ip = f' from {esc(r["source_ip"])}' if isinstance(r["source_ip"], str) else ""
    st.markdown(f'<div class="kv">{esc(str(r["timestamp"])[:19])} · <b>{esc(r["username"])}</b> ran '
                f'<b>{esc(r["event_name"])}</b> on {esc(short(r["target_node"]))}{ip}</div>',
                unsafe_allow_html=True)


def plain_reasons(exp):
    """At most three short sentences: what each model keyed on. Only what pushed the risk UP,
    no attribution units -- the numbers are under "Technical detail"."""
    models, seq, gr = exp["models"], exp.get("sequence"), exp.get("graph")
    s_share, g_share = models["sequence"]["share"], models["graph"]["share"]
    out = []
    if seq and s_share >= MIN_DRIVER_SHARE:
        ups = [f.get("label") or f["feature"] for f in seq.get("top_features", []) if f["contribution_window"] > 0][:2]
        if ups:
            out.append(f"<b>Activity pattern</b> ({s_share:.0%} of the risk): {esc(' and '.join(ups))}.")
        before = [e for e in seq.get("top_events", []) if e["effect"] > 0]
        if before:
            e = before[0]
            out.append(f"<b>What came before</b>: {esc(unseen(e['event_name']))}, {ago(e['minutes_before'])} "
                       f"earlier, made it look more like an attack.")
    if models["graph"]["probability"] is None:
        out.append("<b>Access graph</b>: no score for this kind of access, so the activity pattern decided alone.")
    elif gr and g_share >= MIN_DRIVER_SHARE:
        feats = [f.get("label") or f["feature"] for f in gr.get("top_features", [])][:2]
        out.append(f"<b>Access graph</b> ({g_share:.0%} of the risk): {esc(' and '.join(feats))}.")
    return out


def render_explanation(exp):
    models = exp["models"]
    g, s = models["graph"]["share"], models["sequence"]["share"]
    seg = ""
    if g > 0:
        seg += f'<div class="g" style="width:{g * 100:.1f}%">Access graph {g:.0%}</div>'
    if s > 0:
        seg += f'<div class="s" style="width:{s * 100:.1f}%">Activity pattern {s:.0%}</div>'
    reasons = plain_reasons(exp) or ["No single factor stands out; both models scored it moderately high."]
    st.markdown(f'<div class="share">{seg}</div><div class="summary"><div class="up" style="margin-bottom:6px">'
                f'Why</div>{"<br>".join("• " + x for x in reasons)}</div>', unsafe_allow_html=True)

    with st.expander("Technical detail"):
        render_technical(exp)


def render_technical(exp):
    """The numbers behind the reasons, on one convention: red to the right raised the risk,
    green to the left lowered it."""
    models, seq, gr = exp["models"], exp.get("sequence"), exp.get("graph")
    pg = models["graph"]["probability"]
    st.markdown(
        f'<div class="legend"><span><i class="sw pos"></i>raised the risk</span>'
        f'<span><i class="sw neg"></i>lowered the risk</span><span><i class="sw mag"></i>weight, direction not measured</span>'
        f'<span style="margin-left:auto">'
        f'HGT score {"n/a" if pg is None else f"{pg:.0%}"} · LSTM score {models["sequence"]["probability"]:.0%} · '
        f'risk {exp["risk_score"]:.2f}/10</span></div>', unsafe_allow_html=True)
    col_f, col_e, col_g = st.columns(3, gap="large")
    with col_f:
        feats = [(f.get("label") or f["feature"], f["contribution_window"]) for f in (seq or {}).get("top_features", [])]
        total = sum(abs(v) for _, v in feats) or 1.0
        st.markdown('<div class="th">Activity pattern (LSTM)<small>what pushed its score, % of the influence '
                    'shown</small></div>'
                    + diverging([(n, v / total * 100) for n, v in feats], lambda v: f"{v:+.0f}%"),
                    unsafe_allow_html=True)
    with col_e:
        evs = [(f'{unseen(e["event_name"])} <span class="when">{ago(e["minutes_before"])} before</span>',
                (seq["score"] - e["score_without"]) * 100) for e in (seq or {}).get("top_events", [])]
        st.markdown('<div class="th">Earlier events<small>points each one added to (or took off) the LSTM '
                    'score</small></div>' + diverging(evs, lambda v: f"{v:+.1f} pts", html_labels=True),
                    unsafe_allow_html=True)
    with col_g:
        if gr:
            feats = [(f.get("label") or f["feature"], f["share"] * 100) for f in gr.get("top_features", [])]
            rel = [(f'{r.get("edge_type") or r["log_id"]} <span class="when">{esc(short(r.get("source_node")))} → '
                    f'{esc(short(r.get("target_node")))}</span>', r["share"] * 100)
                   for r in gr.get("related_events", []) if r["share"] >= MIN_RELATED_SHARE]
            st.markdown('<div class="th">Access graph (HGT)<small>how much it weighed each factor (size only, not '
                        'direction)</small></div>'
                        + diverging(feats, lambda v: f"{v:.0f}%", one_sided=True)
                        + ('<div class="th" style="margin-top:14px">Linked events in the graph</div>'
                           + diverging(rel, lambda v: f"{v:.0f}%", one_sided=True, html_labels=True) if rel else ""),
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="th">Access graph (HGT)</div><div class="note">No graph score for this '
                        'event; the LSTM decided alone.</div>', unsafe_allow_html=True)
    st.markdown('<div class="note" style="margin-top:8px">Methods: LSTM features by Integrated Gradients; earlier '
                'events by removing each one and re-scoring; HGT by gradient × input on this event\'s edge. The '
                f'model shares above are exact (risk = w·HGT + (1−w)·LSTM). log_id {esc(exp["log_id"])}</div>',
                unsafe_allow_html=True)


def diverging(items, value_fmt, one_sided=False, html_labels=False):
    """label | bar | value rows, sorted by size. Signed values grow right (red, raised the risk)
    or left (green, lowered it) from a centre line. one_sided is for sizes with no direction
    (the HGT's attribution shares are of |gradient x input|): neutral bars from the left."""
    if not items:
        return '<div class="note">nothing to show</div>'
    items = sorted(items, key=lambda kv: -abs(kv[1]))
    mx = max(abs(v) for _, v in items) or 1.0
    rows = ""
    for name, v in items:
        pos = v >= 0
        width = abs(v) / mx * (100 if one_sided else 50)
        bar = (f'<i class="mag" style="left:0;width:{width:.0f}%"></i>' if one_sided else
               f'<i class="{"pos" if pos else "neg"}" style="{"left" if pos else "right"}:50%;width:{width:.0f}%"></i>')
        rows += (f'<div class="dv"><span class="lab">{name if html_labels else esc(name)}</span>'
                 f'<span class="track{" one" if one_sided else ""}">{bar}</span>'
                 f'<span class="v {"mag" if one_sided else "pos" if pos else "neg"}">{value_fmt(v)}</span></div>')
    return rows


def render_attack_chain(r):
    chain = (f'<span class="cnode"><span class="t">{esc(str(r["username"])[-24:])}</span>'
             f'<span class="s2">acting principal</span></span><span class="cedge">──▶</span>'
             f'<span class="cnode"><span class="t">{esc(r["event_name"])}</span>'
             f'<span class="s2">{esc(r["edge_type"])}</span></span><span class="cedge">──▶</span>'
             f'<span class="cnode"><span class="t">{esc(short(r["target_node"]))}</span>'
             f'<span class="s2">target resource</span></span>')
    st.markdown(f'<div class="card"><div class="hd">Attack chain (from the structural triplet)</div>'
                f'<div style="padding:16px">{chain}</div></div>', unsafe_allow_html=True)


# ── page ─────────────────────────────────────────────────────────────────
live = st.toggle("Live updates", value=True,
                 help=f"Refresh every {REFRESH_SECONDS} s as the pipeline scores new logs. Pause to read calmly.")


@st.fragment(run_every=REFRESH_SECONDS if live else None)
def console():
    threshold = alert_threshold()
    if not os.path.exists(SCORES):
        st.info("Waiting for the pipeline's first scores (output/risk_scores.csv) -- start it with "
                "run.cmd / ./run.sh. This page refreshes on its own.")
        return
    mtime = os.path.getmtime(SCORES)
    df = load_scores(mtime)
    alerts, explanations, by_group = load_alerts()
    is_live = time.time() - mtime < LIVE_WITHIN_SECONDS
    fl = df[df["fast_lane"]]

    st.markdown(
        f"""<div class="topbar"><span class="dot{'' if is_live else ' idle'}"></span>
        <span>User: <b>Admin</b></span><span>|</span>
        <span>Pipeline: <b style="color:{'#3ecf8e' if is_live else '#9aa0b4'}">
        {'Live' if is_live else 'Idle'}</b></span><span>|</span>
        <span>Model: <b>HGT + LSTM ensemble</b>, alert at {threshold:.2f}/10</span>
        <span class="alerts">⚠ {len({a["principal"] for a in alerts})} principals alerted</span></div>""",
        unsafe_allow_html=True)

    for col, val, lab, cls in zip(st.columns(5), (
            f"{len(df):,}", f"{int(df['alert'].sum()):,}", f"{len(fl):,}", f"{len(alerts):,}",
            f"{df['risk_score'].mean():.2f}"), (
            "Events processed", "Alert events", "Fast-lane events", "Alerts raised", "Avg risk /10"),
            ("acc", "crit", "fl", "crit", "acc")):
        col.markdown(f'<div class="stat {cls}">{val}</div><div class="l">{lab}</div>', unsafe_allow_html=True)
    st.write("")

    render_fast_lane(fl)

    # ── processed-log stream ──
    head, pick_filter = st.columns([3, 2])
    head.markdown(f'<div class="card" style="margin-bottom:6px"><div class="hd"><span class="live"></span>'
                  f'Processed logs -- every event, in arrival order; new ones are added at the bottom'
                  f'<span class="sub">click a row to see why</span></div></div>', unsafe_allow_html=True)
    which = pick_filter.radio("Show", list(FILTERS), horizontal=True, label_visibility="collapsed")
    view = df if FILTERS[which] is None else df[df[FILTERS[which]]]
    table = pd.DataFrame({
        "Flag": [flag_of(r) for r in view[["fast_lane", "alert"]].to_dict("records")],
        "Time": view["timestamp"].astype(str).str[:19],
        "Principal": view["username"],
        "Action": view["event_name"],
        "Target": view["target_node"].map(short),
        "HGT": view["p_graph"],
        "LSTM": view["p_sequence"],
        "Risk": view["risk_score"],
        "Why": np.where(view["log_id"].isin(explanations), "explained", ""),
    })
    selection = st.dataframe(
        table, height=TABLE_HEIGHT, hide_index=True, width="stretch",
        on_select="rerun", selection_mode="single-row", key=f"logs_{which}",
        column_config={
            "Flag": st.column_config.TextColumn(width="small"),
            "HGT": st.column_config.NumberColumn(format="%.2f", help="graph model probability (blank: LSTM only)"),
            "LSTM": st.column_config.NumberColumn(format="%.2f", help="sequence model probability"),
            "Risk": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=10,
                                                    help=f"ensemble risk; alert at {threshold:.2f}"),
            "Why": st.column_config.TextColumn(help="an ensemble explanation is available"),
        })

    # ── why was this flagged ──
    rows = selection.selection.rows if selection else []
    if rows and rows[0] < len(view):
        r = view.iloc[rows[0]]
        title = "Why was this event flagged?" if (r["alert"] or r["fast_lane"]) else "Selected event"
    else:
        explained = view[view["log_id"].isin(explanations)]
        if explained.empty:
            st.markdown('<div class="note">Click a row to see its scores. Explanations appear here as soon '
                        'as the first alert is raised.</div>', unsafe_allow_html=True)
            return
        r = explained.iloc[-1]
        title = "Why was this flagged? -- the latest explained alert (click any row to see another)"
    st.markdown(f'<div class="card" style="margin:14px 0 10px"><div class="hd">{esc(title)}</div></div>',
                unsafe_allow_html=True)
    render_event_header(r, threshold)
    exp = explanations.get(r["log_id"])
    if exp:
        render_explanation(exp)
    elif r["alert"] or r["fast_lane"]:
        alert = by_group.get((r["username"], str(r["log_id"]).rsplit(":", 1)[0]))
        msg = "Only the riskiest events of each alert get their own explanation."
        if alert and alert.get("explanations"):
            top = alert["explanations"][0]
            msg += (f" This one is part of an alert on <b>{esc(alert['principal'])}</b> "
                    f"({alert['n_flagged_events']} flagged events). Its top event, {esc(top['event_name'])} "
                    f"({top['risk_score']:.2f}/10), was flagged because:<br>"
                    + "<br>".join("• " + x for x in plain_reasons(top)))
        st.markdown(f'<div class="summary">{msg}</div>', unsafe_allow_html=True)
    render_attack_chain(r)


console()
st.markdown(
    '<div style="color:#646b82;font-size:11px;margin-top:10px">Model: CloudSec-XAI · HGT + LSTM ensemble · '
    'explanations: ensemble_explain.py (exact model shares; LSTM Integrated Gradients + leave-one-event-out; '
    'HGT gradient × input) · MITRE ATT&CK TA0004 / TA0005 / TA0008</div>', unsafe_allow_html=True)
