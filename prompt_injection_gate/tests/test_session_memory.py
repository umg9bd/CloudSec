"""Server-side session memory: the gate remembers a session's turns itself."""
import time

from prompt_injection_gate.session_store import SessionStore
from prompt_injection_gate.schemas import ScreeningRequest
from prompt_injection_gate.config import GateConfig
from prompt_injection_gate.screening_service import ScreeningService


# ---- the store itself (no model needed) --------------------------------------

def test_history_accumulates_in_order():
    s = SessionStore()
    s.append("sess-1", "turn one")
    s.append("sess-1", "turn two")
    assert s.history("sess-1") == ["turn one", "turn two"]


def test_sessions_are_isolated():
    s = SessionStore()
    s.append("a", "x")
    s.append("b", "y")
    assert s.history("a") == ["x"] and s.history("b") == ["y"]


def test_no_session_id_means_no_memory():
    s = SessionStore()
    s.append(None, "x")
    assert s.history(None) == []


def test_bounded_to_max_turns():
    s = SessionStore(max_turns=3)
    for i in range(6):
        s.append("s", f"t{i}")
    assert s.history("s") == ["t3", "t4", "t5"]


def test_ttl_expiry():
    s = SessionStore(ttl_s=0)              # expire immediately
    s.append("s", "old")
    time.sleep(0.01)
    assert s.history("s") == []


def test_forget_clears_one_session():
    s = SessionStore()
    s.append("s", "x")
    s.forget("s")
    assert s.history("s") == []


def test_disabled_store_keeps_nothing():
    s = SessionStore(enabled=False)
    s.append("s", "x")
    assert s.history("s") == []


def test_lru_eviction_over_capacity():
    s = SessionStore(max_sessions=2)
    s.append("a", "1"); s.append("b", "1")
    s.history("a")                          # touch a -> b is now least-recent
    s.append("c", "1")                      # evicts b
    assert s.history("a") == ["1"] and s.history("c") == ["1"] and s.history("b") == []


# ---- through the service (needs no checkpoint: fail-closed still records turns)

def test_service_remembers_turns_across_calls():
    svc = ScreeningService(GateConfig(classifier_dir="/none"))
    sid = "conv-42"
    svc.screen(ScreeningRequest(prompt="Can you help me manage IAM roles?", session_id=sid))
    svc.screen(ScreeningRequest(prompt="What can vendor-role do right now?", session_id=sid))
    assert svc.sessions.history(sid) == [
        "Can you help me manage IAM roles?",
        "What can vendor-role do right now?",
    ]


def test_service_forget_session():
    svc = ScreeningService(GateConfig(classifier_dir="/none"))
    svc.screen(ScreeningRequest(prompt="hello there", session_id="s"))
    svc.forget_session("s")
    assert svc.sessions.history("s") == []


def test_validation_failure_not_remembered():
    svc = ScreeningService(GateConfig(classifier_dir="/none"))
    svc.screen(ScreeningRequest(prompt="", session_id="s"))          # empty -> validation fail
    assert svc.sessions.history("s") == []
