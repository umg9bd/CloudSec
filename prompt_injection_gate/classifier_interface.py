"""
classifier_interface.py
=======================
Component C: the fine-tuned classifier behind a clean, validated interface (spec 6C).

`ClassifierResult` is always a structured object, never free-form text. When the
checkpoint or its thresholds are missing, or inference raises, the result's
`status` is "unavailable"/"error" and `verdict` is UNCERTAIN -- the decision
combiner then fails closed (never SAFE). The transformer stack (torch/
transformers) is imported lazily so importing this module never requires it.

Scores are the model's raw sigmoid outputs; they are NOT treated as calibrated
probabilities. The SAFE/MALICIOUS cut-offs come from gate_model.json, which the
training run derived from the held-out dev slice.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .config import GateConfig
from .schemas import Decision


@dataclass
class ClassifierResult:
    status: str                       # "ok" | "unavailable" | "error"
    verdict: Decision                 # SAFE | MALICIOUS | REVIEW_REQUIRED (UNCERTAIN)
    attack_score: float | None
    category_scores: dict[str, float] = field(default_factory=dict)   # score per category
    flagged_categories: list[str] = field(default_factory=list)       # score >= category threshold
    model_version: str = "unavailable"
    rationale: str = ""
    error: str | None = None


class Classifier:
    """Loads a fine-tuned checkpoint lazily; safe to construct with none present."""

    def __init__(self, cfg: GateConfig):
        self.cfg = cfg
        self._loaded = False
        self._load_error: str | None = None
        self._tok = None
        self._model = None
        self._meta: dict = {}

    @property
    def available(self) -> bool:
        self._ensure()
        return self._loaded

    @property
    def version(self) -> str:
        self._ensure()
        return self._meta.get("model_version", "unavailable")

    def _ensure(self):
        if self._loaded or self._load_error:
            return
        d = self.cfg.classifier_dir
        meta_path = os.path.join(d, "gate_model.json")
        weights = os.path.join(d, "model.safetensors")
        if not (os.path.isfile(meta_path) and os.path.isfile(weights)):
            self._load_error = "checkpoint_not_found"
            return
        try:
            import torch                                      # noqa: F401  (lazy)
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            with open(meta_path, encoding="utf-8") as f:
                self._meta = json.load(f)
            self._tok = AutoTokenizer.from_pretrained(d)
            self._model = AutoModelForSequenceClassification.from_pretrained(d)
            self._model.eval()
            self._loaded = True
        except Exception as e:                                # torch missing, corrupt ckpt, etc.
            self._load_error = f"load_failed:{type(e).__name__}"

    def _encode(self, text: str):
        import torch
        max_len = int(self._meta.get("max_len", 256))
        budget = max_len - 2
        ids = self._tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) > budget:
            ids = ids[:budget // 2] + ids[-(budget - budget // 2):]
        ids = [self._tok.cls_token_id] + ids + [self._tok.sep_token_id]
        return torch.tensor([ids])

    def classify(self, text: str) -> ClassifierResult:
        self._ensure()
        if not self._loaded:
            return ClassifierResult("unavailable", Decision.REVIEW_REQUIRED, None,
                                    error=self._load_error or "unavailable")
        try:
            import torch
            x = self._encode(text)
            with torch.no_grad():
                scores = torch.sigmoid(self._model(input_ids=x,
                                                   attention_mask=torch.ones_like(x)).logits)[0].tolist()
            cats = self._meta["categories"]
            attack = float(scores[0])
            cat_scores = {c: float(scores[1 + i]) for i, c in enumerate(cats)}
            cat_thr = self._meta.get("category_thresholds", {})
            flagged = [c for c in cats
                       if cat_scores[c] >= cat_thr.get(c, {}).get("threshold", 0.5)
                       and cat_thr.get(c, {}).get("calibrated", False)]
            mal = float(self._meta["malicious_threshold"]); safe = float(self._meta["safe_threshold"])
            if attack >= mal:
                verdict = Decision.MALICIOUS
            elif attack < safe:
                verdict = Decision.SAFE
            else:
                verdict = Decision.REVIEW_REQUIRED
            return ClassifierResult("ok", verdict, attack, cat_scores, flagged,
                                    self._meta.get("model_version", "unknown"),
                                    rationale=_rationale(verdict, flagged))
        except Exception as e:
            return ClassifierResult("error", Decision.REVIEW_REQUIRED, None,
                                    model_version=self._meta.get("model_version", "unknown"),
                                    error=f"inference_failed:{type(e).__name__}")


def _rationale(verdict: Decision, flagged: list[str]) -> str:
    if verdict == Decision.SAFE:
        return "classifier attack score below the safe cut-off"
    top = ", ".join(flagged[:3]) if flagged else "no single category above its cut-off"
    return f"classifier attack score in the {verdict.value.lower()} band; categories: {top}"
