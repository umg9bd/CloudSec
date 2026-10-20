"""
Singleton in-memory catalogue loader and verifier for Stage 3.

Guarantees thread-safe, fast loading of the validated AWS action catalogue and fixtures.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional


class CatalogueLoader:
    _instance: Optional["CatalogueLoader"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "CatalogueLoader":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._catalogue_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalogue")
        self._catalogue_file = os.path.join(self._catalogue_dir, "aws_action_catalogue.json")
        self._load_and_validate()
        self._initialized = True

    def _load_and_validate(self) -> None:
        if not os.path.exists(self._catalogue_file):
            raise FileNotFoundError(f"Action catalogue not found at {self._catalogue_file}")

        with open(self._catalogue_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate top-level keys
        for key in ("catalogue_version", "supported_tasks", "security_constraint_triggers", "actions"):
            if key not in data:
                raise ValueError(f"Catalogue missing required top-level key: {key}")

        self._version: str = data["catalogue_version"]
        self._last_reviewed: str = data.get("last_reviewed", "unknown")
        self._supported_tasks: Dict[str, Any] = data["supported_tasks"]
        self._security_triggers: Dict[str, Any] = data["security_constraint_triggers"]
        self._actions: Dict[str, Any] = data["actions"]

        # Validate each action entry
        required_action_keys = (
            "action_id",
            "task_category",
            "service",
            "api_operation",
            "iam_permission",
            "required_parameters",
            "parameter_schema",
            "depends_on",
            "cloudtrail_mapping",
        )
        required_mapping_keys = (
            "mapping_rule_id",
            "eventSource",
            "eventName",
            "request_parameter_template",
            "unknown_fields",
            "source_urls",
            "validation_status",
        )

        for act_id, act in self._actions.items():
            for k in required_action_keys:
                if k not in act:
                    raise ValueError(f"Action '{act_id}' missing required key '{k}'")
            ct_map = act["cloudtrail_mapping"]
            for mk in required_mapping_keys:
                if mk not in ct_map:
                    raise ValueError(f"Action '{act_id}' cloudtrail_mapping missing '{mk}'")
            if ct_map["validation_status"] != "VERIFIED":
                raise ValueError(f"Action '{act_id}' does not have VERIFIED status")

    @property
    def version(self) -> str:
        return self._version

    @property
    def supported_tasks(self) -> List[str]:
        return list(self._supported_tasks.keys())

    def is_task_supported(self, task_category: str) -> bool:
        return task_category in self._supported_tasks

    def get_task_def(self, task_category: str) -> Optional[Dict[str, Any]]:
        return self._supported_tasks.get(task_category)

    def get_action(self, action_id: str) -> Optional[Dict[str, Any]]:
        return self._actions.get(action_id)

    def get_security_trigger(self, trigger_key: str) -> Optional[Dict[str, Any]]:
        return self._security_triggers.get(trigger_key)

    def get_actions_for_task(self, task_category: str) -> List[Dict[str, Any]]:
        task_def = self.get_task_def(task_category)
        if not task_def:
            return []
        actions = []
        for act_id in task_def.get("primary_actions", []):
            act = self.get_action(act_id)
            if act:
                actions.append(act)
        return actions

    def get_fixture_path(self, fixture_ref: str) -> str:
        return os.path.normpath(os.path.join(self._catalogue_dir, fixture_ref))

    def load_fixture(self, fixture_ref: str) -> Optional[Dict[str, Any]]:
        path = self.get_fixture_path(fixture_ref)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None


def get_catalogue() -> CatalogueLoader:
    return CatalogueLoader()
