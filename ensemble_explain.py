"""
ensemble_explain.py -- why did the ensemble flag this event?

The live risk is  risk = w * p_graph + (1 - w) * p_sequence   (p_sequence alone when the
graph model has no weights for the event's relation). This module explains one flagged event
at three levels, always on the exact inputs the pipeline scored it with (the same graph
window, the same per-principal LSTM history), so an explanation matches the score shown:

  1. Which model drove it. The ensemble is linear, so each model's share of the risk is exact:
     w * p_graph / risk  and  (1 - w) * p_sequence / risk.

  2. Graph model (HGT, or whichever graph checkpoint is configured) -- gradient x input on the
     window graph, for the flagged event's logit:
       * top_features: which of the flagged edge's own features pushed it up, grouped by
         explainability.EDGE_FEATURE_NAMES (the one-hot action block counts as "edge_type");
       * related_events: which OTHER events in the window moved it most. Their edge features
         reach the flagged edge through message passing, so a high share means "this earlier
         activity around the same identities is part of why the graph model flagged it".
     Gradient x input is a first-order attribution, reported as shares of the total, not as
     an exact decomposition.

  3. Sequence model (LSTM, whichever checkpoint is configured) -- lstm_explain.explain_seq:
       * top_events: leave-one-event-out over the principal's 10-minute window (logit effect of
         removing each earlier event);
       * top_features: Integrated Gradients per feature (logit units; they sum to the score's
         change from the all-absent baseline).

Nothing here changes a score. It runs only for alerting events (pipeline.py explains the top
few events of each alert), and every function takes model objects rather than checkpoint
paths, so it explains whatever pipeline_config.json serves.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch

from data_loader import scored_edge_types
from explainability import EDGE_FEATURE_NAMES, _feature_groups
import lstm_explain

TOP_K = 5
MIN_RELATED_SHARE = 0.05   # a related event must carry >= 5% of the graph attribution to be named
MIN_DRIVER_SHARE = 0.25    # a model's reasons go in the summary only if it carries >= 25% of the risk

# Plain-English names for the summary line. The JSON keeps each raw feature name next to its
# label, so nothing is lost; an unlisted feature is shown by its raw name.
FEATURE_LABELS = {
    # graph model: edge features (data_loader.EDGE_ATTR_NUMERIC_COLS) + the one-hot action block
    "hop_count": "identity hops from the original principal",
    "privilege_gain": "privilege gained over the role that granted it",
    "privilege_gain_defined": "privilege gain is measurable",
    "action_global_frequency_log": "how common this action is overall",
    "is_privilege_escalation_technique": "known privilege-escalation action",
    "is_read_only": "read-only action",
    "abnormal_path_frequency_rank": "unusual principal-to-target path",
    "edge_type": "the action type itself",
    # sequence model: feature_engine9.TEMPORAL_COLS
    "no_mfa": "no MFA on the session",
    "mfa_absent": "MFA status not recorded",
    "principal_type_prior_risk": "risk learned for this principal type",
    "principal_type_idx": "principal type",
    "has_access_key": "uses an access key",
    "action_velocity": "short gap since this principal's previous action",
    "is_new_action": "first time this principal does this action",
    "session_duration_normalized": "session length",
    "events_per_minute_normalized": "events per minute in the session",
    "time_sin": "time of day", "time_cos": "time of day",
    "is_weekend": "weekend activity", "is_off_hours": "off-hours activity",
    "action_risk_prior": "risk learned for this action",
    "event_name_idx": "the action itself", "event_source_idx": "the AWS service called",
    "is_write_action": "write action", "read_only_absent": "read/write not recorded",
    "has_error": "the call returned an error", "is_access_denied": "access denied",
    "is_iam_event": "IAM call", "is_recon_action": "reconnaissance-style call (Describe/List/Get)",
    "is_defense_evasion": "logging disabled or deleted", "is_get_caller_identity": "GetCallerIdentity check",
    "is_malicious_user_agent": "attack-tool user agent", "is_public_ip": "public source IP",
    "params_length_normalized": "size of the request parameters",
    "targets_sensitive_resource": "targets a sensitive-looking resource",
    "is_non_default_region": "unusual AWS region", "is_create_key": "creates an access key",
    "is_secrets_or_kms": "Secrets Manager / KMS call", "is_permission_modification": "changes permissions",
    "policy_statement_count_normalized": "size of the policy document",
    "has_wildcard_action": "policy grants wildcard actions", "has_wildcard_resource": "policy grants wildcard resources",
    "privileged_action_reach": "policy reaches privileged actions",
    "new_permission_count_log": "number of permissions newly granted",
    "permission_expansion_score": "share of AWS actions newly granted",
    "privilege_delta": "increase in the highest access level held",
    "target_permission_coverage": "permissions held by the identity acted on",
    "actor_permission_coverage": "permissions the actor is known to hold",
    "principal_handoff": "identity obtained from another principal (role assumption / issued keys)",
    "causal_depth_normalized": "identity-chain depth",
    "lineage_enabling_steps_normalized": "recent identity-enabling steps along the chain",
    "pe_write_recent": "a privilege-escalation write just before",
    "log_secs_since_pe": "time since the last privilege-escalation write",
    "log_seconds_since_prev": "gap since the previous event",
}


def label(feature: str) -> str:
    """Plain-English name of a graph or LSTM feature (event_name=X -> 'the action X')."""
    if feature.startswith("event_name="):
        return f"the action {feature.split('=', 1)[1]}"
    return FEATURE_LABELS.get(feature, feature)


# ── 2. graph model ──────────────────────────────────────────────────────────

def explain_graph_events(model: torch.nn.Module, data, log_ids_in_order: List[str],
                         targets: Iterable[str], structural: Optional[pd.DataFrame] = None,
                         top_k: int = TOP_K) -> Dict[str, dict]:
    """Per target log_id: the flagged edge's own feature shares and the most influential
    other events in the window. `data` / `log_ids_in_order` come from
    GNNScorer.build_graph, i.e. exactly what the model scored (log_ids in output order).
    `structural` (optional) supplies source/target/action for related events."""
    position = {lid: i for i, lid in enumerate(log_ids_in_order)}
    triples = [t for t in scored_edge_types(data) if t in getattr(model, "edge_types", scored_edge_types(data))]
    where = {}                                  # log_id -> (triple, local index)
    for t in triples:
        for i, lid in enumerate(data[t].log_id):
            where[str(lid)] = (t, i)
    info = {}
    if structural is not None and len(structural):
        info = structural.drop_duplicates("log_id").set_index("log_id")[
            ["source_node", "target_node", "edge_type"]].to_dict("index")

    model.eval()
    out = {}
    originals = {t: data[t].edge_attr for t in data.edge_types}
    try:
        for lid in targets:
            lid = str(lid)
            if lid not in position or lid not in where:
                continue                        # not scored by the graph model (untrained relation)
            for t in data.edge_types:
                data[t].edge_attr = originals[t].detach().clone().requires_grad_(True)
            logits = model(data)
            logit = logits[position[lid]]
            model.zero_grad(set_to_none=True)
            logit.backward()

            triple, local = where[lid]
            own = (data[triple].edge_attr.grad[local] * data[triple].edge_attr[local]).abs().detach().cpu().numpy()
            own_groups = [(name, float(own[cols].sum()))
                          for name, cols in _feature_groups(len(own), EDGE_FEATURE_NAMES)]
            own_total = sum(v for _, v in own_groups) or 1e-12

            influence = []
            for t in triples:
                g = data[t].edge_attr.grad
                if g is None:
                    continue
                per_edge = (g * data[t].edge_attr).abs().sum(-1).detach().cpu().numpy()
                influence += [(str(other), float(v)) for other, v in zip(data[t].log_id, per_edge)]
            total = sum(v for _, v in influence) or 1e-12
            related = sorted(((o, v) for o, v in influence if o != lid and v > 0), key=lambda x: -x[1])

            out[lid] = {
                "relation": triple[1],
                "source_type": triple[0], "target_type": triple[2],
                "logit": round(float(logit.detach()), 4),
                "top_features": [{"feature": n, "share": round(v / own_total, 4)}
                                 for n, v in sorted(own_groups, key=lambda x: -x[1])[:top_k]],
                "own_edge_share": round(dict(influence).get(lid, 0.0) / total, 4),
                "related_events": [dict({"log_id": o, "share": round(v / total, 4)},
                                        **{k: info.get(o, {}).get(k) for k in ("edge_type", "source_node", "target_node")})
                                   for o, v in related[:top_k]],
                "method": "gradient x input (first-order; shares of total attribution)",
            }
    finally:
        for t, attr in originals.items():
            data[t].edge_attr = attr
    return out


# ── 3. sequence model ───────────────────────────────────────────────────────

def explain_sequence_events(model: torch.nn.Module, ckpt: dict, vocab: dict, seqs,
                            targets: Iterable[str], device, top_k: int = TOP_K) -> Dict[str, dict]:
    """Per target log_id: lstm_explain.explain_seq on the sequence the pipeline scored."""
    names = lstm_explain.feature_names(ckpt)
    id2name = lstm_explain.id_to_name(vocab)
    by_id = {str(s.log_id): s for s in seqs}
    out = {}
    for lid in targets:
        seq = by_id.get(str(lid))
        if seq is not None:
            out[str(lid)] = lstm_explain.explain_seq(model, seq, names, id2name, device, top_k)
    return out


# ── 1. combine ──────────────────────────────────────────────────────────────

def _model_shares(p_graph, p_sequence: float, weight_graph: float) -> dict:
    graph_scored = p_graph is not None and not (isinstance(p_graph, float) and math.isnan(p_graph))
    g = weight_graph * float(p_graph) if graph_scored else 0.0
    s = (1.0 - weight_graph) * float(p_sequence) if graph_scored else float(p_sequence)
    risk = g + s
    return {
        "graph": {"probability": None if not graph_scored else round(float(p_graph), 4),
                  "weight": weight_graph if graph_scored else 0.0,
                  "contribution": round(g, 4), "share": round(g / risk, 4) if risk else 0.0},
        "sequence": {"probability": round(float(p_sequence), 4),
                     "weight": (1.0 - weight_graph) if graph_scored else 1.0,
                     "contribution": round(s, 4), "share": round(s / risk, 4) if risk else 0.0},
        "graph_scored": graph_scored,
        "risk": round(risk, 4),
    }


def _lstm_reasons(seq_exp: Optional[dict], k: int = 2) -> List[str]:
    if not seq_exp:
        return []
    feats = [f for f in seq_exp.get("top_features", []) if f["contribution_window"] > 0][:k]
    events = [e for e in seq_exp.get("top_events", []) if e["effect"] > 0][:k]
    out = [f"{label(f['feature'])} ({f['contribution_window']:+.2f})" for f in feats]
    out += [f"earlier {event_label(e['event_name'])} {e['minutes_before']:g} min before ({e['effect']:+.2f})"
            for e in events]
    return out


def event_label(event_name: str) -> str:
    """An earlier event's name as the LSTM saw it: actions outside its training vocabulary all
    map to <UNK>, which means nothing to an analyst."""
    return "an action unseen in training" if event_name in ("<UNK>", "<PAD>", None) else event_name


def _graph_reasons(graph_exp: Optional[dict], k: int = 2) -> List[str]:
    if not graph_exp:
        return []
    out = [f"{label(f['feature'])} ({f['share']:.0%})" for f in graph_exp.get("top_features", [])[:k]]
    out += [f"related {r.get('edge_type') or r['log_id']} ({r['share']:.0%})"
            for r in graph_exp.get("related_events", [])[:1] if r["share"] >= MIN_RELATED_SHARE]
    return out


def combine(event: dict, graph_exp: Optional[dict], seq_exp: Optional[dict], weight_graph: float,
            threshold: float, fast_lane_reason: Optional[str] = None,
            graph_unavailable: Optional[str] = None) -> dict:
    """One flagged event's ensemble explanation. `event` needs log_id, event_name, username,
    timestamp, p_graph, p_sequence (the pipeline's scored row). `graph_unavailable` says why the
    graph model has no score when that is not the usual "relation unseen in training"."""
    shares = _model_shares(event.get("p_graph"), event["p_sequence"], weight_graph)
    for exp in (graph_exp, seq_exp):      # label every reported feature in the JSON too
        for f in (exp or {}).get("top_features", []):
            f.setdefault("label", label(f["feature"]))
    driver = "graph" if shares["graph"]["share"] >= shares["sequence"]["share"] else "sequence"
    reasons = {"graph": _graph_reasons(graph_exp), "sequence": _lstm_reasons(seq_exp)}
    parts = [f"risk {shares['risk'] * 10:.2f}/10 (alert at {threshold * 10:.2f})"]
    for name, model_name in (("sequence", "LSTM"), ("graph", "HGT")):
        if name == "graph" and not shares["graph_scored"]:
            parts.append(f"HGT: {graph_unavailable or 'relation not seen in training'}, LSTM only")
            continue
        head = f"{model_name} {shares[name]['share']:.0%} (p={shares[name]['probability']:.2f})"
        if shares[name]["share"] < MIN_DRIVER_SHARE:
            parts.append(f"{head}: not a driver")
            continue
        why = "; ".join(reasons[name]) if reasons[name] else "no single dominant factor"
        parts.append(f"{head}: {why}")
    if fast_lane_reason:
        parts.insert(0, f"FAST-LANE rule: {fast_lane_reason}")
    return {
        "log_id": str(event["log_id"]),
        "event_name": event.get("event_name"),
        "principal": event.get("username"),
        "timestamp": str(event.get("timestamp")),
        "risk_score": round(shares["risk"] * 10, 2),
        "threshold": threshold,
        "driven_by": driver,
        "models": {k: shares[k] for k in ("graph", "sequence")},
        "graph": graph_exp,
        "sequence": seq_exp,
        "fast_lane": fast_lane_reason,
        "summary": " | ".join(parts),
    }
