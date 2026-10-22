"""
rule_engine.py
==============
Component B: deterministic checks from the versioned rules catalogue (spec 6B).

The engine loads rules_catalogue.yaml once, matches each rule against the folded
view + normalization signals + decoded segments, and returns the findings with
the evidence for each. It never authorises a prompt: an empty result is "no rule
fired", which the decision combiner must not read as SAFE on its own.

Matching is structural (regex, word-class co-occurrence, normalization flags).
Word-class tokens are ordinary English; a co-occurrence rule fires on their
arrangement, not on any single token, and defensive framing downgrades a finding.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import yaml

from .config import GateConfig
from .normalization import Normalized, Decoded

_OUTBOUND = re.compile(r"(https?://|www\.|\b[\w.+-]+@[\w-]+\.[\w.-]+|webhook|pastebin|requestbin|\bcurl\b|\bexfil)", re.I)
_WORD = re.compile(r"[a-z0-9_]+(?:\s+prompt|\s+message|\s+instruction)?")


@dataclass
class Finding:
    rule_id: str
    category: str
    severity: str
    disposition: str          # "block" | "review"
    evidence: str             # short, structural; never the full prompt
    channel: str              # "user_prompt" or a context source
    defensive_context: bool   # defensive framing seen nearby (downgrades weight)
    technique_id: str | None = None


class RuleEngine:
    def __init__(self, cfg: GateConfig):
        self.cfg = cfg
        with open(cfg.rules_catalogue_path, encoding="utf-8") as f:
            cat = yaml.safe_load(f)
        self.version = cat.get("version", "unknown")
        self.rules = cat["rules"]
        self.tc = {k: [t.lower() for t in v] for k, v in cat.get("token_classes", {}).items()}
        # pre-compile regex rules
        self._re = {r["id"]: re.compile(r["signal"]["pattern"])
                    for r in self.rules if r["signal"].get("type") == "regex"}

    # -- helpers --------------------------------------------------------------
    def _positions(self, tokens: list[str], cls: str) -> list[int]:
        want = set(self.tc.get(cls, []))
        multi = [w for w in want if " " in w]
        pos = [i for i, t in enumerate(tokens) if t in want]
        if multi:                                   # handle 2-word class members
            joined = " ".join(tokens)
            for phrase in multi:
                if phrase in joined:
                    pos.append(0)                   # presence is enough for these
        return pos

    def _cooccur(self, tokens, a, b, window, also=None, also_window=0) -> bool:
        pa, pb = self._positions(tokens, a), self._positions(tokens, b)
        if not (pa and pb):
            return False
        near = any(abs(i - j) <= window for i in pa for j in pb)
        if not near:
            return False
        if also:
            pc = self._positions(tokens, also)
            anchor = [j for i in pa for j in pb if abs(i - j) <= window]
            return any(abs(k - m) <= also_window for k in pc for m in anchor) if pc else False
        return True

    def _defensive(self, tokens) -> bool:
        return bool(self._positions(tokens, "defensive_frame"))

    def _injection_structure(self, folded: str) -> bool:
        toks = folded.split()
        return self._cooccur(toks, "override_verb", "instruction_noun", 6, "prior_scope", 10) \
            or self._cooccur(toks, "reveal_verb", "secret_noun", 8)

    # -- main -----------------------------------------------------------------
    def evaluate(self, norm: Normalized, decoded: list[Decoded], channel: str = "user_prompt") -> list[Finding]:
        folded = norm.folded
        toks = folded.split()
        sig = norm.signals
        defensive = self._defensive(toks)
        out: list[Finding] = []

        def add(rule, evidence):
            out.append(Finding(rule["id"], rule["category"], rule["severity"],
                               rule.get("disposition", "review"), evidence, channel, defensive,
                               (rule.get("technique") or {}).get("id")))

        for rule in self.rules:
            s = rule["signal"]; t = s.get("type")
            if t == "normalization_flag":
                if sig.get(s["flag"]):
                    add(rule, f"{s['flag']}=true")
            elif t == "normalization_count":
                if sig.get(s["field"], 0) >= s.get("min", 1):
                    add(rule, f"{s['field']}={sig.get(s['field'])}")
            elif t == "regex":
                # case-preserving view so uppercase format ids (AKIA...) survive
                if self._re[rule["id"]].search(norm.cleaned):
                    add(rule, f"matched {rule['id']} pattern")
            elif t == "cooccurrence":
                if self._cooccur(toks, s["a"], s["b"], s["window"],
                                 s.get("require_also"), s.get("also_window", 0)):
                    add(rule, f"{s['a']}+{s['b']} within {s['window']} tokens")
            elif t == "cooccurrence_three":
                if self._cooccur(toks, s["a"], s["b"], s["window"]) and _OUTBOUND.search(folded):
                    add(rule, f"{s['a']}+{s['b']}+outbound destination")
            elif t == "decoded_contains_override":
                for d in decoded:
                    if self._injection_structure(re.sub(r"\s+", " ", d.plaintext).strip().lower()):
                        add(rule, f"{d.kind} segment decodes to instruction structure")
                        break
            elif t == "indirect_channel_instruction":
                if channel != "user_prompt" and self._injection_structure(folded):
                    add(rule, f"instruction structure in {channel}")
            # fake-verdict handled by its regex rule above
        return out
