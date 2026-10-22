"""
schema_validation.py
====================
Validates the module's three JSON Schemas (request, response, catalogue) with no
third-party dependency.

WHY THIS EXISTS
---------------
`jsonschema` is not in requirements.txt and the fast test tier must run with
nothing but the Python environment. The schemas in schemas/ are plain JSON Schema
(draft 2020-12 vocabulary), so downstream teams can validate against them with any
library; this file implements the subset THESE schemas use:

    type, enum, const, required, properties, additionalProperties, items,
    minItems, maxItems, minLength, maxLength, pattern, minimum, maximum,
    $ref (local "#/$defs/..."), allOf, anyOf, oneOf, if/then/else

A keyword outside that subset raises SchemaError instead of being silently
ignored, so extending a schema with an unsupported keyword is caught immediately
rather than weakening validation. tests/test_security_hardening_schemas.py
cross-checks this validator against `jsonschema` (when installed) on a corpus of
valid and invalid documents.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"

_ANNOTATIONS = {"$schema", "$id", "title", "description", "$defs", "definitions", "default", "examples", "$comment"}
_HANDLED = {"type", "enum", "const", "required", "properties", "additionalProperties", "items", "minItems",
            "maxItems", "minLength", "maxLength", "pattern", "minimum", "maximum", "$ref", "allOf", "anyOf",
            "oneOf", "if", "then", "else"}


class SchemaError(Exception):
    """The schema itself uses something this validator does not implement."""


_cache: Dict[str, Dict[str, Any]] = {}


def load_schema(name: str) -> Dict[str, Any]:
    """Load schemas/<name>.schema.json (cached)."""
    if name not in _cache:
        _cache[name] = json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))
    return _cache[name]


def _type_ok(value: Any, type_name: str) -> bool:
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "null":
        return value is None
    raise SchemaError(f"unsupported type {type_name!r}")


def _resolve(root: Dict[str, Any], ref: str) -> Dict[str, Any]:
    if not ref.startswith("#/"):
        raise SchemaError(f"only local $ref values are supported, got {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def validate(instance: Any, schema: Dict[str, Any], root: Optional[Dict[str, Any]] = None, path: str = "$") -> List[str]:
    """Return a list of error strings (empty = valid)."""
    root = schema if root is None else root
    unknown = set(schema) - _HANDLED - _ANNOTATIONS
    if unknown:
        raise SchemaError(f"unsupported schema keyword(s) at {path}: {sorted(unknown)}")
    errors: List[str] = []
    if "$ref" in schema:
        merged = {**_resolve(root, schema["$ref"]), **{k: v for k, v in schema.items() if k != "$ref"}}
        return validate(instance, merged, root, path)

    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_type_ok(instance, t) for t in types):
            return [f"{path}: expected {'|'.join(types)}, got {type(instance).__name__}"]
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} is not one of {schema['enum']}")
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: must equal {schema['const']!r}")

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']} characters")
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            errors.append(f"{path}: does not match pattern {schema['pattern']!r}")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: above maximum {schema['maximum']}")

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: needs at least {schema['minItems']} item(s)")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: allows at most {schema['maxItems']} item(s)")
        if "items" in schema:
            for i, item in enumerate(instance):
                errors.extend(validate(item, schema["items"], root, f"{path}[{i}]"))

    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required property {key!r}")
        props = schema.get("properties", {})
        for key, value in instance.items():
            if key in props:
                errors.extend(validate(value, props[key], root, f"{path}.{key}"))
            elif "additionalProperties" in schema:
                extra = schema["additionalProperties"]
                if extra is False:
                    errors.append(f"{path}: unexpected property {key!r}")
                elif isinstance(extra, dict):
                    errors.extend(validate(value, extra, root, f"{path}.{key}"))

    for sub in schema.get("allOf", []):
        errors.extend(validate(instance, sub, root, path))
    if "anyOf" in schema and not any(not validate(instance, s, root, path) for s in schema["anyOf"]):
        errors.append(f"{path}: matches none of the anyOf alternatives")
    if "oneOf" in schema:
        matches = sum(1 for s in schema["oneOf"] if not validate(instance, s, root, path))
        if matches != 1:
            errors.append(f"{path}: must match exactly one oneOf alternative (matched {matches})")
    if "if" in schema:
        branch = "then" if not validate(instance, schema["if"], root, path) else "else"
        if branch in schema:
            errors.extend(validate(instance, schema[branch], root, path))
    return errors


def validate_request(document: Any) -> List[str]:
    return validate(document, load_schema("hardening_request"))


def validate_response(document: Any) -> List[str]:
    return validate(document, load_schema("hardening_response"))


def validate_catalogue_structure(document: Any) -> List[str]:
    return validate(document, load_schema("control_catalogue"))
