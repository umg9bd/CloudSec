"""
Explainability for the LSTM + Transformer (any checkpoint with the v5 model class: the live v5
`lstm_transformer_clean` and v6.3 both load here).

Per alert (local), for the event the model scored:
  - top_events:   leave-one-event-out over the 10-min window -- drop each earlier event and re-score;
                  effect = logit(with) - logit(without), so it stays informative when the score is
                  saturated near 1. "CreateAccessKey 3 min ago added +2.1 to the logit".
  - top_features: Integrated Gradients (zero baseline = the "feature absent" value the model is trained
                  with via feature dropout; the event-name embedding goes to the <UNK>/PAD row, which is
                  zero). Contributions are in logit units and sum to logit(x) - logit(baseline)
                  (`ig_completeness_error` reports how closely).
Global (dev only):
  - permutation importance: shuffle one feature (or the event name) across all real-dev events and
    measure the drop in session / event AUC-PR.

Real test data is never read here.

From the repo root:
  python temporal-analysis/lstm_explain.py --model v6.3 --alerts 5        # explain the top dev alerts
  python temporal-analysis/lstm_explain.py --model v5 --alerts 5 --global  # + permutation importance
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

import train_lstm_transformer as v5  # noqa: E402

MODELS = {
    "v5": next((p for p in (ROOT / "artifacts" / d / "temporal_lstm_transformer.pt"
                            for d in ("lstm_transformer_clean", "lstm_transformer_v5_live")) if p.exists()),
               ROOT / "artifacts" / "lstm_transformer_clean" / "temporal_lstm_transformer.pt"),
    "v6.3": ROOT / "artifacts" / "lstm_transformer_v6_3" / "temporal_lstm_transformer.pt",
}
DT_FEATURE = "log_seconds_since_prev"  # the extra column add_extra_feats appends
MAX_DT = math.log1p(3600.0)
IG_STEPS = 32


def feature_names(ckpt: dict) -> list[str]:
    return list(ckpt["feature_cols"]) + [DT_FEATURE]


def id_to_name(vocab: dict[str, int]) -> dict[int, str]:
    out: dict[int, str] = {}
    for name, i in vocab.items():
        out.setdefault(int(i), name)  # first name wins when two share an id
    out[0] = "<UNK>"
    return out


def _tensors(seq, device):
    return (torch.as_tensor(np.array(seq.event_idxs), dtype=torch.long, device=device).unsqueeze(0),
            torch.as_tensor(np.array(seq.feats), dtype=torch.float32, device=device).unsqueeze(0),
            torch.as_tensor([seq.length], dtype=torch.long))


@torch.no_grad()
def _prob(model, idx, feats, lengths) -> np.ndarray:
    return torch.sigmoid(model(idx, feats, lengths)).cpu().numpy()


@torch.no_grad()
def _logit(model, idx, feats, lengths) -> np.ndarray:
    return model(idx, feats, lengths).cpu().numpy()


def drop_event(idx: np.ndarray, feats: np.ndarray, length: int, j: int, dt_col: int):
    """Window without event j (right-padded again); the next event's gap becomes the sum of both gaps."""
    keep = [k for k in range(length) if k != j]
    f = feats[:length].copy()
    if j + 1 < length:
        f[j + 1, dt_col] = 0.0 if j == 0 else min(
            math.log1p(math.expm1(f[j, dt_col]) + math.expm1(f[j + 1, dt_col])), MAX_DT)
    new_idx = np.zeros_like(idx)
    new_feats = np.zeros_like(feats)
    new_idx[: length - 1] = idx[keep]
    new_feats[: length - 1] = f[keep]
    return new_idx, new_feats


def event_effects(model, seq, dt_col: int, device) -> list[dict]:
    """Leave-one-event-out for every earlier event in the window (the scored event itself is kept)."""
    L = int(seq.length)
    if L < 2:
        return []
    idx, feats = np.array(seq.event_idxs), np.array(seq.feats)
    variants = [drop_event(idx, feats, L, j, dt_col) for j in range(L - 1)]
    bi = torch.as_tensor(np.stack([v[0] for v in variants]), dtype=torch.long, device=device)
    bf = torch.as_tensor(np.stack([v[1] for v in variants]), dtype=torch.float32, device=device)
    l_without = _logit(model, bi, bf, torch.full((L - 1,), L - 1, dtype=torch.long))
    l_with = float(_logit(model, *_tensors(seq, device))[0])
    sig = lambda x: 1.0 / (1.0 + math.exp(-x))
    gaps = np.expm1(feats[:L, dt_col])  # seconds since the previous event
    return [{"position": j, "event_idx": int(idx[j]),
             "minutes_before": round(float(gaps[j + 1: L].sum()) / 60.0, 2),
             "effect": round(l_with - float(l_without[j]), 4),
             "score_without": round(sig(float(l_without[j])), 4)} for j in range(L - 1)]


def integrated_gradients(model, seq, device, steps: int = IG_STEPS) -> dict:
    """IG from (zero features, zero name embedding) to the real window, midpoint rule, one batch."""
    idx, feats, lengths = _tensors(seq, device)
    alphas = (torch.arange(steps, dtype=torch.float32, device=device) + 0.5) / steps
    b_idx = idx.expand(steps, -1)
    b_len = lengths.expand(steps)
    b_feats = (alphas.view(-1, 1, 1) * feats).requires_grad_(True)
    store = {}

    def hook(_module, _inp, out):
        scaled = out * alphas.view(-1, 1, 1)
        scaled.retain_grad()
        store["emb"] = scaled
        return scaled

    h = model.embedding.register_forward_hook(hook)
    try:
        with torch.backends.cudnn.flags(enabled=False):  # cuDNN RNN backward needs train mode
            logits = model(b_idx, b_feats, b_len)
        logits.sum().backward()
    finally:
        h.remove()
    with torch.no_grad():
        emb_full = model.embedding(idx)[0]
        attr_feats = (feats[0] * b_feats.grad.mean(0)).cpu().numpy()          # (T, F)
        attr_name = (emb_full * store["emb"].grad.mean(0)).sum(-1).cpu().numpy()  # (T,)
        logit_x = float(model(idx, feats, lengths)[0])
    logit_0 = _baseline_logit(model, idx, feats, lengths)
    model.zero_grad(set_to_none=True)
    L = int(seq.length)
    total = float(attr_feats[:L].sum() + attr_name[:L].sum())
    return {"feats": attr_feats[:L], "name": attr_name[:L], "logit": logit_x, "baseline_logit": logit_0,
            "completeness_error": abs(total - (logit_x - logit_0))}


@torch.no_grad()
def _baseline_logit(model, idx, feats, lengths) -> float:
    h = model.embedding.register_forward_hook(lambda _m, _i, out: out * 0.0)
    try:
        return float(model(idx, torch.zeros_like(feats), lengths)[0])
    finally:
        h.remove()


def explain_seq(model, seq, names: list[str], id2name: dict[int, str], device, top_k: int = 5) -> dict:
    """Why did the model score this event the way it did? JSON-ready."""
    model.eval()
    dt_col = names.index(DT_FEATURE)
    L = int(seq.length)
    idx, feats = np.array(seq.event_idxs), np.array(seq.feats)
    ig = integrated_gradients(model, seq, device)
    window_tot = ig["feats"].sum(0)
    feats_out = [{"feature": names[k], "value_now": round(float(feats[L - 1, k]), 4),
                  "contribution_now": round(float(ig["feats"][L - 1, k]), 4),
                  "contribution_window": round(float(window_tot[k]), 4)} for k in range(len(names))]
    feats_out.append({"feature": f"event_name={id2name.get(int(idx[L - 1]), '?')}", "value_now": None,
                      "contribution_now": round(float(ig["name"][L - 1]), 4),
                      "contribution_window": round(float(ig["name"].sum()), 4)})
    feats_out.sort(key=lambda r: -abs(r["contribution_window"]))
    events = event_effects(model, seq, dt_col, device)
    for e in events:
        e["event_name"] = id2name.get(e.pop("event_idx"), "?")
    events.sort(key=lambda r: -abs(r["effect"]))
    return {
        "log_id": seq.log_id, "username": seq.username, "timestamp": str(seq.timestamp),
        "event_name": id2name.get(int(idx[L - 1]), "?"), "score": round(1 / (1 + math.exp(-ig["logit"])), 4),
        "window_events": L,
        "top_events": events[:top_k],
        "top_features": feats_out[:top_k],
        "baseline_score": round(1 / (1 + math.exp(-ig["baseline_logit"])), 4),
        "ig_completeness_error": round(ig["completeness_error"], 4),
    }


def explain_text(e: dict) -> str:
    lines = [f"ALERT {e['event_name']} by {e['username']} at {e['timestamp']}  score={e['score']:.3f}"]
    if e["top_events"]:
        lines.append("  earlier events (logit effect | score if removed):")
        lines += [f"    {r['event_name']:<28} {r['minutes_before']:>5.1f} min before  {r['effect']:+.3f} | {r['score_without']:.3f}"
                  for r in e["top_events"]]
    lines.append("  features (IG, logit units, whole window | current event):")
    lines += [f"    {r['feature']:<36} {r['contribution_window']:+.3f} | {r['contribution_now']:+.3f}"
              + (f"   (now={r['value_now']:g})" if r["value_now"] is not None else "") for r in e["top_features"]]
    return "\n".join(lines)


@torch.no_grad()
def _batched_probs(model, idx, feats, lengths, device, bs: int = 2048) -> np.ndarray:
    out = []
    for i in range(0, len(idx), bs):
        out.append(_prob(model, idx[i:i + bs].to(device), feats[i:i + bs].to(device), lengths[i:i + bs]))
    return np.concatenate(out)


def permutation_importance(model, dev, names: list[str], device, seed: int = 42) -> pd.DataFrame:
    """Drop in real-dev session / event AUC-PR when one feature (or the event name) is shuffled across all
    real events. Shuffles valid (non-pad) positions only."""
    model.eval()
    rng = np.random.default_rng(seed)
    idx = torch.as_tensor(np.stack([s.event_idxs for s in dev.seqs]), dtype=torch.long)
    feats = torch.as_tensor(np.stack([s.feats for s in dev.seqs]), dtype=torch.float32)
    lengths = torch.as_tensor([s.length for s in dev.seqs], dtype=torch.long)
    y_e = np.array([s.label for s in dev.seqs])
    log_ids = [s.log_id for s in dev.seqs]
    valid = v5.valid_mask(lengths, idx.size(1))
    y_s = dev.sessions.to_numpy()

    def aps(p):
        return average_precision_score(y_s, dev.session_scores(log_ids, p)), average_precision_score(y_e, p)

    base_s, base_e = aps(_batched_probs(model, idx, feats, lengths, device))
    rows = []
    for k, name in enumerate(names + ["event_name"]):
        f2, i2 = feats.clone(), idx.clone()
        if name == "event_name":
            vals = i2[valid]
            i2[valid] = vals[torch.as_tensor(rng.permutation(len(vals)))]
        else:
            vals = f2[..., k][valid]
            f2[..., k][valid] = vals[torch.as_tensor(rng.permutation(len(vals)))]
        s_ap, e_ap = aps(_batched_probs(model, i2, f2, lengths, device))
        rows.append({"feature": name, "session_ap_drop": round(base_s - s_ap, 4), "event_ap_drop": round(base_e - e_ap, 4)})
    df = pd.DataFrame(rows).sort_values("session_ap_drop", ascending=False).reset_index(drop=True)
    df.attrs.update(base_session_ap=round(base_s, 4), base_event_ap=round(base_e, 4))
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="v6.3", help="v5 | v6.3 | path to a checkpoint")
    ap.add_argument("--alerts", type=int, default=5,
                    help="explain the top-scored event of each of the N highest-scored real-dev attack sessions")
    ap.add_argument("--global", dest="global_", action="store_true", help="also run permutation importance on dev")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    import os
    import train_lstm_transformer_v6_3 as t63

    path = Path(MODELS.get(args.model, args.model)).resolve()
    out_dir = path.parent / "explanations"
    out_dir.mkdir(exist_ok=True)
    os.chdir(REPO)  # feature_engine9's paths are repo-relative
    device = torch.device("cpu")
    model, ckpt, _ = v5.load_checkpoint(path, device=device)
    model.eval()
    names = feature_names(ckpt)
    assert len(names) == model.tab_head[-1].in_features, (len(names), model.tab_head[-1].in_features)
    vocab = dict(ckpt["event_name_vocab"])
    dev = t63.RealDev.build(t63.real_dev_features(), vocab, list(ckpt["feature_cols"]))

    ev = v5.score_seqs(model, dev.seqs, device)
    ev["session_id"] = [dev.event_sid[i] for i in ev["log_id"]]
    ev["session_label"] = [dev.sessions[s] for s in ev["session_id"]]
    thr = float(ckpt.get("event_threshold") or ckpt.get("threshold") or 0.5)
    top = ev[ev["session_label"] == 1].sort_values("P_event", ascending=False).drop_duplicates("session_id")
    alerts = top.head(args.alerts)
    by_id = {s.log_id: s for s in dev.seqs}
    id2name = id_to_name(vocab)
    exps = [explain_seq(model, by_id[i], names, id2name, device, args.top_k) for i in alerts["log_id"]]
    print(f"model {path.relative_to(REPO)}  (event threshold {thr:.3f}; real dev only)\n")
    for e in exps:
        print(explain_text(e) + "\n")
    (out_dir / "dev_alerts.json").write_text(json.dumps(exps, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'dev_alerts.json'}")

    if args.global_:
        pi = permutation_importance(model, dev, names, device)
        pi.to_csv(out_dir / "permutation_importance.csv", index=False)
        print(f"\npermutation importance (dev session AUC-PR {pi.attrs['base_session_ap']}, "
              f"event AUC-PR {pi.attrs['base_event_ap']}):")
        print(pi.head(12).to_string(index=False))
        print(f"wrote {out_dir / 'permutation_importance.csv'}")


if __name__ == "__main__":
    main()
