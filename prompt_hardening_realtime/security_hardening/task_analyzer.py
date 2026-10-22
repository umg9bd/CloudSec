"""
task_analyzer.py
================
Reads an upstream-approved prompt and produces a `TaskProfile`: what the task is, which
approved services and resources it touches, what is exposed, what sensitive data is
involved, which insecure configurations the prompt ASKS FOR, and which security
requirements it ALREADY states.

WHY THIS EXISTS
---------------
The control catalogue decides which controls apply from named *facts* (models.FACT_VOCABULARY).
Something has to turn free text into those facts. Plain keyword hits are not enough, so this
module adds the three things that make the facts trustworthy:

  * NEGATION.  "Do not make the bucket public" must not become a request for a public bucket
    (it would raise a bogus SECURITY_CONFLICT). A match is dropped when its clause negates it;
    a match that itself starts with a negator ("no authentication", "without encryption") is
    treated as a request. See `_is_negated`.
  * RELATIONSHIPS.  "Take the username from the query string and look it up in the users table"
    is untrusted input reaching a SQL sink. Sources and sinks are detected separately; a flow is
    *stated* when both sit in one sentence and *inferred* (recorded in `TaskProfile.inferred`)
    when the task is an endpoint/web app that receives client input and also has the sink.
  * SCOPE.  Services are classified with config/scope.yaml. Python's `lambda x: ...` is not AWS
    Lambda (the bare word needs AWS context); "don't use KMS" is not a dependency on KMS; and an
    unknown service is only reported when an ARN or `<service>.amazonaws.com` names it.

Patterns are DATA (config/lexicon.yaml); stated-requirement cues come from the catalogue so
the analyser and the catalogue's `satisfied_by` rules cannot drift apart. `load_lexicon()`
rejects an inconsistent lexicon (unknown fact/value, uncompilable regex, a vocabulary value with
no pattern, a scope service with no entry) instead of silently never matching.

WHAT THIS IS NOT
----------------
It is a heuristic over English text, not a parser, and recall on phrasing it has never seen is
unmeasured (no corpus of real prompts exists in this repository). It never decides that a prompt
is safe, never contacts AWS, and adds no service support: detecting the NAME of an unsupported
service exists only so the module can say it will not handle it.

Standard library + PyYAML only.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Pattern, Sequence, Set, Tuple

import yaml

from .catalogue import Catalogue
from .config import ScopeConfig, load_scope
from .models import FACT_VOCABULARY, ServiceFindings, TaskProfile, canonical_sha256, empty_facts

LEXICON_PATH = Path(__file__).resolve().parent / "config" / "lexicon.yaml"

MAX_SNIPPETS = 3
MAX_SNIPPET_CHARS = 160
MAX_ENTITIES = 10

# 1:1 character substitutions only, so every match index stays valid in the original text.
_NORMALISE = str.maketrans({
    "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
    "\u00a0": " ", "\u2013": "-", "\u2014": "-", "\u2011": "-",
})
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[`])|\n+")
_REGION_LABEL = re.compile(r"^[a-z]{2}(?:-[a-z]+)+-\d+$")
_AWS_ACCESS_KEY_ID = re.compile(r"\bAKIA[0-9A-Z]{16}\b")

# Sections where "derived" or "catalogue-sourced" means no per-value patterns are expected.
_NO_PATTERN_FACTS = {"input_flows": "derived from input_flows.sources + sinks", "stated_requirements": "catalogue requirement_cues"}


class LexiconError(ValueError):
    """The lexicon is malformed or inconsistent with the vocabulary / scope; `problems` lists everything."""

    def __init__(self, problems: List[str]):
        self.problems = list(problems)
        shown = "; ".join(self.problems[:8]) + (f" (+{len(self.problems) - 8} more)" if len(self.problems) > 8 else "")
        super().__init__(f"invalid lexicon: {shown}")


# ── lexicon model ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Pattern:
    regex: Pattern[str]
    negation_guard: bool = True
    requires_aws_context: bool = False


@dataclass(frozen=True)
class _Negation:
    boundary: Pattern[str]
    pre_window_words: int
    pre: Tuple[Pattern[str], ...]
    inside: Tuple[Pattern[str], ...]
    start_negator: Pattern[str]
    stated_disabling: Pattern[str]


@dataclass(frozen=True)
class Lexicon:
    version: str
    sha256: str
    negation: _Negation
    aws_context: Tuple[Pattern[str], ...]
    services: Dict[str, Tuple[_Pattern, ...]]
    arn_re: Pattern[str]
    endpoint_re: Pattern[str]
    implied_services: Dict[str, str]
    facts: Dict[str, Dict[str, Tuple[_Pattern, ...]]]
    flow_sources: Tuple[Pattern[str], ...]
    implicit_source_contexts: FrozenSet[str]
    flow_sinks: Dict[str, Tuple[Pattern[str], ...]]


def _compile(text: Any, where: str, problems: List[str]) -> Optional[Pattern[str]]:
    if not isinstance(text, str) or not text:
        problems.append(f"{where}: pattern must be a non-empty string")
        return None
    try:
        return re.compile(text, re.IGNORECASE)
    except re.error as exc:
        problems.append(f"{where}: bad regex {text[:60]!r}: {exc}")
        return None


def _patterns(items: Any, where: str, default_guard: bool, problems: List[str], aws: bool = False) -> Tuple[_Pattern, ...]:
    out: List[_Pattern] = []
    if not isinstance(items, list):
        problems.append(f"{where}: must be a list")
        return ()
    for i, item in enumerate(items):
        guard = default_guard
        if isinstance(item, Mapping):
            guard = bool(item.get("negation_guard", default_guard))
            item = item.get("regex")
        compiled = _compile(item, f"{where}[{i}]", problems)
        if compiled is not None:
            out.append(_Pattern(compiled, guard, aws))
    return tuple(out)


def parse_lexicon(data: Mapping[str, Any], scope: ScopeConfig) -> Lexicon:
    """Validate a lexicon document against the vocabulary and scope, then compile it."""
    problems: List[str] = []

    neg_raw = data.get("negation") or {}
    neg = _Negation(
        boundary=_compile(neg_raw.get("clause_boundary"), "negation.clause_boundary", problems) or re.compile(r"[.]"),
        pre_window_words=int(neg_raw.get("pre_window_words", 6)),
        pre=tuple(p for i, s in enumerate(neg_raw.get("pre", [])) if (p := _compile(s, f"negation.pre[{i}]", problems))),
        inside=tuple(p for i, s in enumerate(neg_raw.get("inside", [])) if (p := _compile(s, f"negation.inside[{i}]", problems))),
        start_negator=_compile(neg_raw.get("start_negator"), "negation.start_negator", problems) or re.compile(r"^$"),
        stated_disabling=_compile(neg_raw.get("stated_disabling"), "negation.stated_disabling", problems) or re.compile(r"^$"),
    )
    aws_ctx = tuple(p for i, s in enumerate(data.get("aws_context", [])) if (p := _compile(s, f"aws_context[{i}]", problems)))

    # services
    services: Dict[str, Tuple[_Pattern, ...]] = {}
    for key, entry in (data.get("services") or {}).items():
        plain = _patterns(entry.get("patterns", []), f"services.{key}.patterns", True, problems)
        weak = _patterns(entry.get("patterns_requiring_aws_context", []), f"services.{key}.patterns_requiring_aws_context", True, problems, aws=True)
        services[key] = plain + weak
        if not services[key]:
            problems.append(f"services.{key}: no patterns")
    required_service_keys = set(scope.vocabulary_services) | set(FACT_VOCABULARY["services"])
    for key in sorted(required_service_keys - set(services)):
        problems.append(f"services: scope/vocabulary service {key!r} has no lexicon entry")

    implied = dict(data.get("implied_services") or {})
    for kind, service in implied.items():
        if kind not in FACT_VOCABULARY["resource_kinds"]:
            problems.append(f"implied_services: unknown resource kind {kind!r}")
        if service not in FACT_VOCABULARY["services"]:
            problems.append(f"implied_services.{kind}: {service!r} is not an approved service in the vocabulary")

    # facts
    facts: Dict[str, Dict[str, Tuple[_Pattern, ...]]] = {}
    section_raw = data.get("facts") or {}
    for fact, body in section_raw.items():
        if fact not in FACT_VOCABULARY:
            problems.append(f"facts: unknown fact {fact!r}")
            continue
        if fact == "stated_requirements":
            problems.append("facts.stated_requirements: cues live in the catalogue (requirement_cues), not the lexicon")
            continue
        default_guard = bool(body.get("negation_guard", True))
        values = body.get("values") or {}
        facts[fact] = {}
        for value, items in values.items():
            if value not in FACT_VOCABULARY[fact]:
                problems.append(f"facts.{fact}: value {value!r} is not in the vocabulary")
                continue
            facts[fact][value] = _patterns(items, f"facts.{fact}.{value}", default_guard, problems)
        if not body.get("derived"):
            for missing in sorted(FACT_VOCABULARY[fact] - set(facts[fact])):
                problems.append(f"facts.{fact}: vocabulary value {missing!r} has no patterns (add them, or the value can never be detected)")
    for fact in FACT_VOCABULARY:
        if fact not in section_raw and fact not in _NO_PATTERN_FACTS and fact != "services":
            problems.append(f"facts: vocabulary fact {fact!r} has no section")

    # input flows
    flow = data.get("input_flows") or {}
    sources = tuple(p for i, s in enumerate(flow.get("sources", [])) if (p := _compile(s, f"input_flows.sources[{i}]", problems)))
    if not sources:
        problems.append("input_flows.sources: needs at least one pattern")
    sinks: Dict[str, Tuple[Pattern[str], ...]] = {}
    for sink, items in (flow.get("sinks") or {}).items():
        if sink not in FACT_VOCABULARY["input_flows"] or sink == "untrusted_input_present":
            problems.append(f"input_flows.sinks: {sink!r} is not a sink flow in the vocabulary")
            continue
        sinks[sink] = tuple(p for i, s in enumerate(items) if (p := _compile(s, f"input_flows.sinks.{sink}[{i}]", problems)))
    for missing in sorted(FACT_VOCABULARY["input_flows"] - {"untrusted_input_present"} - set(sinks)):
        problems.append(f"input_flows.sinks: flow {missing!r} has no sink patterns")
    implicit = frozenset(flow.get("implicit_source_contexts", []))
    for ctx in implicit - FACT_VOCABULARY["app_context"]:
        problems.append(f"input_flows.implicit_source_contexts: {ctx!r} is not an app_context value")

    arn_re = _compile(data.get("arn_pattern"), "arn_pattern", problems)
    endpoint_re = _compile(data.get("endpoint_pattern"), "endpoint_pattern", problems)
    if problems:
        raise LexiconError(problems)
    return Lexicon(
        version=str(data.get("lexicon_version", "")), sha256=canonical_sha256(data), negation=neg, aws_context=aws_ctx, services=services,
        arn_re=arn_re, endpoint_re=endpoint_re, implied_services=implied, facts=facts,
        flow_sources=sources, implicit_source_contexts=implicit, flow_sinks=sinks,
    )


def load_lexicon(path: Optional[Path] = None, scope: Optional[ScopeConfig] = None) -> Lexicon:
    path = Path(path) if path else LEXICON_PATH
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise LexiconError([f"cannot read {path}: {exc}"]) from exc
    if not isinstance(data, dict):
        raise LexiconError([f"{path}: top level must be a mapping"])
    return parse_lexicon(data, scope or load_scope())


# ── per-prompt working state ────────────────────────────────────────────────

class _Doc:
    """Normalised text plus the lookup tables every detector shares."""

    def __init__(self, raw: str, lex: Lexicon):
        self.text = raw.translate(_NORMALISE)
        spans: List[Tuple[int, int]] = []
        pos = 0
        for m in _SENTENCE_SPLIT.finditer(self.text):
            spans.append((pos, m.start()))
            pos = m.end()
        spans.append((pos, len(self.text)))
        self.spans = [(a, b) for a, b in spans if b > a] or [(0, len(self.text))]
        self._starts = [a for a, _ in self.spans]
        self.boundary_ends = [m.end() for m in lex.negation.boundary.finditer(self.text)]
        self.aws_ctx = any(p.search(self.text) for p in lex.aws_context)

    def sentence_index(self, pos: int) -> int:
        return max(0, bisect.bisect_right(self._starts, pos) - 1)

    def excerpt(self, pos: int) -> str:
        a, b = self.spans[self.sentence_index(pos)]
        snippet = re.sub(r"\s+", " ", self.text[a:b]).strip()
        return snippet if len(snippet) <= MAX_SNIPPET_CHARS else snippet[: MAX_SNIPPET_CHARS - 3].rstrip() + "..."

    def clause_start(self, pos: int) -> int:
        i = bisect.bisect_right(self.boundary_ends, pos) - 1
        return self.boundary_ends[i] if i >= 0 else 0


class TaskAnalyzer:
    """Stateless between calls: construct once, call `analyze` per prompt."""

    def __init__(self, catalogue: Catalogue, scope: Optional[ScopeConfig] = None, lexicon: Optional[Lexicon] = None):
        self.catalogue = catalogue
        self.scope = scope or load_scope()
        self.lex = lexicon or load_lexicon(scope=self.scope)

    # ── negation ────────────────────────────────────────────────────────────
    def _is_negated(self, doc: _Doc, start: int, end: int, stated: bool = False) -> bool:
        neg = self.lex.negation
        # An audit/remediation sentence can mention an insecure configuration without requesting it.
        sentence_start, sentence_end = doc.spans[doc.sentence_index(start)]
        sentence = doc.text[sentence_start:sentence_end]
        if re.search(r"\b(?:find|identify|detect|audit|scan|check|review)\b.*\b(?:tighten|restrict|remediate|fix|close|correct|remove)\b", sentence, re.I):
            return True
        pre = " ".join(doc.text[doc.clause_start(start):start].split()[-neg.pre_window_words:])
        if any(p.search(pre) for p in neg.pre):
            return True
        matched = doc.text[start:end]
        starts_with_negator = bool(neg.start_negator.match(matched))
        if not starts_with_negator and any(p.search(matched) for p in neg.inside):
            return True
        if stated and (neg.stated_disabling.search(pre) or (not starts_with_negator and neg.stated_disabling.search(matched))):
            return True
        return False

    def _scan(self, doc: _Doc, patterns: Iterable[_Pattern], stated: bool = False, limit: int = MAX_SNIPPETS,
              use_guard: bool = True) -> List[Tuple[int, int]]:
        hits: List[Tuple[int, int]] = []
        for pat in patterns:
            if pat.requires_aws_context and not doc.aws_ctx:
                continue
            for m in pat.regex.finditer(doc.text):
                if m.end() == m.start():
                    continue
                if use_guard and pat.negation_guard and self._is_negated(doc, m.start(), m.end(), stated):
                    continue
                hits.append((m.start(), m.end()))
                if len(hits) >= limit:
                    return hits
        return hits

    # ── main entry ──────────────────────────────────────────────────────────
    def analyze(self, prompt: str, task_metadata: Optional[Mapping[str, Any]] = None,
                known_constraints: Optional[Mapping[str, Any]] = None) -> TaskProfile:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        meta = dict(task_metadata or {})
        known = dict(known_constraints or {})
        doc = _Doc(prompt, self.lex)
        facts = empty_facts()
        evidence: Dict[str, Dict[str, List[str]]] = {}
        inferred: Set[str] = set()
        notes: List[str] = []
        findings = ServiceFindings()

        def note(fact: str, value: str, text: str) -> None:
            bucket = evidence.setdefault(fact, {}).setdefault(value, [])
            if text and text not in bucket and len(bucket) < MAX_SNIPPETS:
                bucket.append(text)

        def add(fact: str, value: str, snippet: str, *, inferred_flag: bool = False) -> None:
            facts[fact].add(value)
            note(fact, value, snippet)
            if inferred_flag:
                inferred.add(f"{fact}:{value}")

        self._detect_services(doc, meta, findings, facts, add, notes)
        for fact, values in self.lex.facts.items():
            for value, patterns in values.items():
                for start, end in self._scan(doc, patterns):
                    # A credential mentioned solely as an object that must NOT be logged is
                    # not evidence that the task handles/stores credentials. Keep logging
                    # policy cues separate from secret-management applicability.
                    if fact == "sensitive_data" and value == "credentials":
                        sentence_start = max(doc.text.rfind(".", 0, start), doc.text.rfind(";", 0, start),
                                             doc.text.rfind("\n", 0, start)) + 1
                        sentence_end_candidates = [pos for pos in (doc.text.find(".", end),
                                                                  doc.text.find(";", end),
                                                                  doc.text.find("\n", end)) if pos >= 0]
                        sentence_end = min(sentence_end_candidates) if sentence_end_candidates else len(doc.text)
                        sentence = doc.text[sentence_start:sentence_end]
                        logging_guard = re.search(
                            r"\b(?:never|do not|don['’]?t|avoid|without|no)\b[^.]{0,80}\b(?:log|logging|record|recording)\b",
                            sentence, re.I)
                        if logging_guard and start >= sentence_start + logging_guard.start():
                            continue
                    add(fact, value, doc.excerpt(start))
        self._declared_metadata(meta, known, facts, add, notes)

        # services implied by a resource kind that only exists in one approved service
        for kind in sorted(facts["resource_kinds"]):
            service = self.lex.implied_services.get(kind)
            if service and service not in facts["services"]:
                add("services", service, (evidence["resource_kinds"][kind] or [""])[0], inferred_flag=True)
                if service not in findings.approved:
                    findings.approved.append(service)

        self._detect_flows(doc, facts, evidence, add)
        self._detect_stated_requirements(doc, facts, add)
        self._literal_secrets(doc, facts, add, notes)
        entities = self._entities(doc)

        signal = any(facts[f] for f in ("services", "resource_kinds", "action_classes", "app_context",
                                        "exposure", "permission_shapes", "credential_handling", "languages"))
        signal = signal or bool(findings.excluded or findings.out_of_scope or findings.recognised)
        words = re.findall(r"[A-Za-z0-9_#+.\-]+", doc.text)
        has_target = any(facts[f] for f in ("services", "resource_kinds", "app_context", "permission_shapes", "exposure"))
        generic_access_only = (not has_target and bool(re.search(r"\b(?:permissions?|access|roles?|polic(?:y|ies)|privileges?)\b", doc.text, re.I)))
        unclear = len(words) < 3 or generic_access_only or (len(words) < 6 and not signal)

        findings.approved = sorted(set(findings.approved))
        return TaskProfile(
            facts=facts, evidence=evidence, entities=entities, services=findings,
            intent_unclear=unclear, inferred=inferred, notes=notes,
        )

    # ── services ────────────────────────────────────────────────────────────
    def _classify_into(self, key: str, findings: ServiceFindings) -> str:
        bucket = self.scope.classify(key)
        target = {"approved": findings.approved, "recognised": findings.recognised,
                  "excluded": findings.excluded, "out_of_scope": findings.out_of_scope}[bucket]
        if key not in target:
            target.append(key)
        return bucket

    def _detect_services(self, doc: _Doc, meta: Mapping[str, Any], findings: ServiceFindings,
                         facts: Dict[str, Set[str]], add, notes: List[str]) -> None:
        for key, patterns in self.lex.services.items():
            bucket = self.scope.classify(key)
            # "don't use KMS" is not a dependency on KMS, so excluded / out-of-scope mentions are
            # negation-guarded; an approved service named in a prohibition is still the topic.
            hits = self._scan(doc, patterns, limit=1, use_guard=(bucket != "approved"))
            if hits:
                self._classify_into(key, findings)
                if bucket == "approved":
                    add("services", key, doc.excerpt(hits[0][0]))
        # explicit identifiers may name a service the lexicon has never heard of
        for regex, label in ((self.lex.arn_re, "ARN"), (self.lex.endpoint_re, "endpoint")):
            for m in regex.finditer(doc.text):
                key = m.group(1).lower()
                if _REGION_LABEL.match(key) or key in ("amazonaws", "aws"):
                    continue
                if self._is_negated(doc, m.start(), m.end()):
                    continue
                bucket = self._classify_into(key, findings)
                if bucket == "approved":
                    add("services", key, doc.excerpt(m.start()))
        declared = meta.get("services") or []
        for raw in declared if isinstance(declared, list) else []:
            if not isinstance(raw, str) or not raw.strip():
                continue
            key = raw.strip().lower().removesuffix(".amazonaws.com")
            bucket = self._classify_into(key, findings)
            if bucket == "approved":
                add("services", key, f"task_metadata.services: {key}")

    # ── caller-declared metadata ────────────────────────────────────────────
    def _declared_metadata(self, meta: Mapping[str, Any], known: Mapping[str, Any],
                           facts: Dict[str, Set[str]], add, notes: List[str]) -> None:
        languages = meta.get("languages") or []
        for lang in languages if isinstance(languages, list) else []:
            key = str(lang).strip().lower()
            if key in FACT_VOCABULARY["languages"]:
                add("languages", key, f"task_metadata.languages: {key}")
            else:
                notes.append(f"task_metadata.languages: {str(lang)!r} is not a language the analyser models; ignored.")
        frameworks = meta.get("frameworks") or []
        if isinstance(frameworks, list) and frameworks:
            joined = " ".join(str(f) for f in frameworks)
            fdoc = _Doc(joined, self.lex)
            for fact in ("languages", "app_context"):
                for value, patterns in self.lex.facts.get(fact, {}).items():
                    if self._scan(fdoc, patterns, limit=1):
                        add(fact, value, f"task_metadata.frameworks: {joined[:80]}")

        env = str(known.get("environment", "")).strip().lower()
        if env in ("production", "prod"):
            add("high_impact", "production_target", "known_constraints.environment: production")
        elif env and env not in ("dev", "development", "test", "testing", "staging", "stage", "sandbox"):
            notes.append(f"known_constraints.environment: {known.get('environment')!r} not recognised; no environment facts derived.")
        cls = str(known.get("data_classification", "")).strip().lower()
        if cls:
            mapped = [value for value, pat in (
                ("personal_data", r"pii|personal"), ("financial", r"financ|pci|payment"),
                ("health", r"health|phi|hipaa|medical"), ("confidential", r"confidential|restricted|secret|internal|sensitive|proprietary"),
            ) if re.search(pat, cls)]
            for value in mapped:
                add("sensitive_data", value, f"known_constraints.data_classification: {cls}")
            if not mapped and cls not in ("public", "none", "unclassified"):
                notes.append(f"known_constraints.data_classification: {known.get('data_classification')!r} not recognised; no sensitivity derived.")

    # ── input flows (source -> sink) ────────────────────────────────────────
    def _detect_flows(self, doc: _Doc, facts: Dict[str, Set[str]], evidence: Dict[str, Dict[str, List[str]]], add) -> None:
        source_spans = [(m.start(), m.end()) for p in self.lex.flow_sources for m in p.finditer(doc.text) if m.end() > m.start()]
        source_sentences = {doc.sentence_index(s) for s, _ in source_spans}
        implicit = sorted(facts["app_context"] & self.lex.implicit_source_contexts)
        receives_input = bool(source_spans) or bool(implicit)
        for sink, patterns in self.lex.flow_sinks.items():
            first_hit: Optional[int] = None
            stated = False
            for pat in patterns:
                for m in pat.finditer(doc.text):
                    if m.end() == m.start():
                        continue
                    if self._is_negated(doc, m.start(), m.end()):
                        continue
                    first_hit = m.start() if first_hit is None else first_hit
                    if doc.sentence_index(m.start()) in source_sentences:
                        first_hit, stated = m.start(), True
                        break
                if stated:
                    break
            if first_hit is None:
                continue
            if stated:
                add("input_flows", sink, doc.excerpt(first_hit))
            elif receives_input:
                add("input_flows", sink, doc.excerpt(first_hit), inferred_flag=True)
        if receives_input or facts["input_flows"]:
            if source_spans:
                snippet = doc.excerpt(source_spans[0][0])
            else:
                snippet = (evidence.get("app_context", {}).get(implicit[0]) or [""])[0] if implicit else ""
            add("input_flows", "untrusted_input_present", snippet, inferred_flag=not source_spans)

    # ── stated requirements (cues come from the catalogue) ──────────────────
    def _detect_stated_requirements(self, doc: _Doc, facts: Dict[str, Set[str]], add) -> None:
        for tag, patterns in self.catalogue.cues.items():
            for pat in patterns:
                for m in pat.finditer(doc.text):
                    if m.end() == m.start():
                        continue
                    if not self._is_negated(doc, m.start(), m.end(), stated=True):
                        add("stated_requirements", tag, doc.excerpt(m.start()))
                        break
                if tag in facts["stated_requirements"]:
                    break

    # ── literal secrets pasted into the prompt ──────────────────────────────
    def _literal_secrets(self, doc: _Doc, facts: Dict[str, Set[str]], add, notes: List[str]) -> None:
        match = _AWS_ACCESS_KEY_ID.search(doc.text)
        if match:
            add("sensitive_data", "credentials", doc.excerpt(match.start()))
            add("credential_handling", "handles_secrets", doc.excerpt(match.start()))
            notes.append("The prompt itself contains what looks like an AWS access key ID; treat it as exposed "
                         "(rotate it) and do not repeat it. This module did not store or use it.")

    # ── literal entities (never invented) ───────────────────────────────────
    def _entities(self, doc: _Doc) -> Dict[str, List[str]]:
        def collect(patterns: Sequence[str], group: int = 1) -> List[str]:
            found: List[str] = []
            for pat in patterns:
                for m in re.finditer(pat, doc.text, re.I):
                    value = m.group(group).strip("`\"'.,;)")
                    if value and value not in found:
                        found.append(value)
                    if len(found) >= MAX_ENTITIES:
                        return found
            return found

        out = {
            "buckets": collect([r"\bs3://([a-z0-9][a-z0-9.\-]{1,61}[a-z0-9])", r"arn:aws[a-z-]*:s3:::([a-z0-9][a-z0-9.\-]{1,61}[a-z0-9])",
                                r"\bbuckets?\s+(?:named|called)\s+[`\"']?([a-z0-9][a-z0-9.\-]{2,62})"]),
            "roles": collect([r"arn:aws[a-z-]*:iam::\d*:role/([\w+=,.@/\-]+)", r"\broles?\s+(?:named|called)\s+[`\"']?([\w+=,.@\-]{1,64})"]),
            "users": collect([r"arn:aws[a-z-]*:iam::\d*:user/([\w+=,.@/\-]+)", r"\b(?:iam\s+)?users?\s+(?:named|called)\s+[`\"']?([\w+=,.@\-]{1,64})"]),
            "cidrs": collect([r"\b(\d{1,3}(?:\.\d{1,3}){3}/\d{1,2})\b", r"(::/0)"]),
            "ports": collect([r"\bports?\s+(\d{2,5})\b", r"\b(\d{2,5})/(?:tcp|udp)\b"]),
            "account_ids": collect([r"(?<![\w-])(\d{12})(?![\w-])"]),
            "arns": collect([r"(arn:aws[a-z-]*:[a-z0-9-]+:[a-z0-9-]*:\d*:[\w:/.\-*+=,@]+)"]),
            "iam_actions": collect([r"\b((?:iam|s3|ec2|sts|ssm|rds|cloudtrail|elasticloadbalancing):[A-Za-z*][A-Za-z0-9*]*)"]),
        }
        return {k: v for k, v in out.items() if v}

