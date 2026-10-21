"""
conditions.py
=============
The small declarative condition language the catalogue uses for "when does this
control apply", "is it already satisfied" and "does this request conflict".

WHY THIS EXISTS
---------------
Applicability must live in the catalogue (data), not in code, so a control can be
added or retuned without a release - and it must be checkable at load time. One
module defines the grammar, validates it against models.FACT_VOCABULARY and
evaluates it, so the loader and the selector can never disagree about what a
condition means.

Grammar (YAML/JSON):
    {fact: <name>, any_of: [v, ...]}    the fact's values intersect the list
    {fact: <name>, all_of: [v, ...]}    the fact's values include every listed value
    {fact: <name>, none_of: [v, ...]}   the fact's values contain none of the list
    {fact: <name>, present: true|false} the fact has (or has no) values at all
    {all: [cond, ...]}   {any: [cond, ...]}   {not: cond}
"""

from __future__ import annotations

from typing import Any, List, Mapping, Set

from .models import FACT_VOCABULARY

_OPERATORS = ("any_of", "all_of", "none_of", "present")
_COMBINATORS = ("all", "any", "not")


def validate_condition(cond: Any, path: str = "condition") -> List[str]:
    """Return a list of problems (empty = well-formed and every fact/value is in the vocabulary)."""
    if not isinstance(cond, Mapping):
        return [f"{path}: must be a mapping"]
    keys = set(cond)
    combinator = keys & set(_COMBINATORS)
    if combinator:
        if len(keys) != 1:
            return [f"{path}: a combinator ({', '.join(sorted(combinator))}) must be the only key"]
        key = next(iter(combinator))
        if key == "not":
            return validate_condition(cond["not"], f"{path}.not")
        items = cond[key]
        if not isinstance(items, list) or not items:
            return [f"{path}.{key}: needs a non-empty list"]
        problems: List[str] = []
        for i, sub in enumerate(items):
            problems.extend(validate_condition(sub, f"{path}.{key}[{i}]"))
        return problems
    fact = cond.get("fact")
    if fact not in FACT_VOCABULARY:
        return [f"{path}: unknown fact {fact!r}"]
    operators = keys - {"fact"}
    if len(operators) != 1 or not operators <= set(_OPERATORS):
        return [f"{path}: needs exactly one of {list(_OPERATORS)} next to 'fact'"]
    op = next(iter(operators))
    if op == "present":
        return [] if isinstance(cond["present"], bool) else [f"{path}.present: must be true or false"]
    values = cond[op]
    if not isinstance(values, list) or not values:
        return [f"{path}.{op}: needs a non-empty list"]
    bad = [v for v in values if v not in FACT_VOCABULARY[fact]]
    return [f"{path}: value(s) {bad} are not valid for fact {fact!r}"] if bad else []


def evaluate_condition(cond: Mapping[str, Any], facts: Mapping[str, Set[str]]) -> bool:
    """Evaluate a condition that has already passed validate_condition()."""
    if "all" in cond:
        return all(evaluate_condition(c, facts) for c in cond["all"])
    if "any" in cond:
        return any(evaluate_condition(c, facts) for c in cond["any"])
    if "not" in cond:
        return not evaluate_condition(cond["not"], facts)
    have = set(facts.get(cond["fact"], ()))
    if "any_of" in cond:
        return bool(have & set(cond["any_of"]))
    if "all_of" in cond:
        return set(cond["all_of"]) <= have
    if "none_of" in cond:
        return not (have & set(cond["none_of"]))
    return bool(have) == cond["present"]


def referenced_values(cond: Mapping[str, Any], fact: str) -> Set[str]:
    """Every value of `fact` that the condition mentions (used to check requirement tags have cue patterns)."""
    found: Set[str] = set()
    if "all" in cond or "any" in cond:
        for sub in cond.get("all", cond.get("any", [])):
            found |= referenced_values(sub, fact)
    elif "not" in cond:
        found |= referenced_values(cond["not"], fact)
    elif cond.get("fact") == fact:
        for op in ("any_of", "all_of", "none_of"):
            found |= set(cond.get(op, ()))
    return found
