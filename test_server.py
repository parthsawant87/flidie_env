# test_server.py — PROJECT ROOT
# Integration tests — requires server running at localhost:8000
# Start server: uvicorn server.app:app --port 8000
# Run tests:    python3 -m pytest test_server.py -v

import pytest
import requests

BASE = "http://localhost:8000"


def _reset(task_id: str = "financial_optimize") -> tuple:
    """Helper: reset and return (session_id, obs_dict)."""
    r = requests.post(f"{BASE}/reset", json={"task_id": task_id})
    assert r.status_code == 200
    sid = r.headers.get("X-Session-ID")
    assert sid, "X-Session-ID header missing from /reset response"
    return sid, r.json()


def _step(sid: str, action: dict) -> dict:
    """Helper: step with session_id, return result dict."""
    r = requests.post(f"{BASE}/step", json=action, headers={"X-Session-ID": sid})
    assert r.status_code == 200, f"Step failed: {r.text}"
    return r.json()


# ── HEALTH ───────────────────────────────────────────────────────────────────

def test_health_returns_ok():
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_tasks_returns_all_three():
    r = requests.get(f"{BASE}/tasks")
    assert r.status_code == 200
    ids = [t["id"] for t in r.json()["tasks"]]
    for expected in ["financial_optimize", "tax_planning", "startup_compliance"]:
        assert expected in ids, f"Missing task: {expected}"


# ── RESET ────────────────────────────────────────────────────────────────────

def test_reset_returns_session_id_header():
    r = requests.post(f"{BASE}/reset", json={"task_id": "financial_optimize"})
    assert r.status_code == 200
    assert "X-Session-ID" in r.headers


def test_reset_returns_valid_observation_fields():
    _, obs = _reset()
    for field in ["scenario_id", "title", "context", "options", "done", "step_count"]:
        assert field in obs, f"Missing field in observation: {field}"


def test_reset_done_is_false():
    _, obs = _reset()
    assert obs["done"] == False


def test_reset_options_has_four_choices():
    _, obs = _reset()
    assert len(obs["options"]) == 4, f"Expected 4 options, got {len(obs['options'])}"
    ids = [o["id"] for o in obs["options"]]
    for expected in ["A", "B", "C", "D"]:
        assert expected in ids


def test_reset_invalid_task_returns_400():
    r = requests.post(f"{BASE}/reset", json={"task_id": "nonexistent_task"})
    assert r.status_code == 400


# ── SECURITY: GROUND TRUTH MUST NEVER APPEAR ─────────────────────────────────

def test_reset_response_has_no_ground_truth():
    """CRITICAL: ground_truth, optimal_option, outcome_map must never appear in HTTP responses."""
    r = requests.post(f"{BASE}/reset", json={"task_id": "financial_optimize"})
    body = r.text
    for forbidden in ["ground_truth", "optimal_option", "outcome_map"]:
        assert forbidden not in body, f"SECURITY VIOLATION: '{forbidden}' found in /reset response"


def test_step_response_has_no_ground_truth():
    sid, _ = _reset()
    result = _step(sid, {"action_type": "ask_clarification", "question_text": "What is the tax bracket?"})
    body = str(result)
    for forbidden in ["ground_truth", "optimal_option", "outcome_map"]:
        assert forbidden not in body, f"SECURITY VIOLATION: '{forbidden}' in /step response"


def test_state_response_has_no_ground_truth():
    sid, _ = _reset()
    r = requests.get(f"{BASE}/state", headers={"X-Session-ID": sid})
    body = r.text
    for forbidden in ["ground_truth", "optimal_option", "outcome_map"]:
        assert forbidden not in body, f"SECURITY VIOLATION: '{forbidden}' in /state response"


# ── STEP ─────────────────────────────────────────────────────────────────────

def test_step_calculate_correct_returns_positive_reward():
    sid, _ = _reset("financial_optimize")
    result = _step(sid, {
        "action_type":    "calculate",
        "expression":    "90000 * 0.3",
        "expected_result": 27000
    })
    assert result["reward"] > 0, f"Correct calc should give positive reward, got {result['reward']}"
    assert result["done"] == False


def test_step_choose_option_terminates_episode():
    sid, _ = _reset()
    result = _step(sid, {"action_type": "choose_option", "option_id": "A"})
    assert result["done"] == True
    assert -1.0 <= result["reward"] <= 1.0


def test_step_after_done_returns_400():
    sid, _ = _reset()
    _step(sid, {"action_type": "choose_option", "option_id": "B"})
    r = requests.post(f"{BASE}/step",
        json={"action_type": "ask_clarification", "question_text": "test?"},
        headers={"X-Session-ID": sid})
    assert r.status_code == 400


def test_step_missing_session_returns_400():
    r = requests.post(f"{BASE}/step",
        json={"action_type": "choose_option", "option_id": "A"})
    assert r.status_code == 400


def test_step_invalid_action_returns_422():
    sid, _ = _reset()
    r = requests.post(f"{BASE}/step",
        json={"bad_field": "nonsense"},
        headers={"X-Session-ID": sid})
    assert r.status_code == 422


# ── TRAJECTORY DEPTH — OBSERVATIONS CHANGE AFTER EACH ACTION ────────────────

def test_calculations_done_accumulates_in_observation():
    """
    After calling calculate(), the observation's calculations_done list
    must grow. This proves trajectory-dependent observation state.
    """
    sid, obs0 = _reset()
    assert obs0["calculations_done"] == []

    result1 = _step(sid, {
        "action_type":     "calculate",
        "expression":     "90000 * 0.3",
        "expected_result": 27000
    })
    calcs = result1["observation"]["calculations_done"]
    assert len(calcs) == 1, "calculations_done should have 1 entry"
    assert calcs[0]["expression"] == "90000 * 0.3"
    assert calcs[0]["result"] == 27000.0


def test_compliance_flags_accumulates_in_observation():
    """After flag_compliance_risk(), the observation's compliance_flags list grows."""
    sid, _ = _reset("startup_compliance")
    result = _step(sid, {
        "action_type":     "flag_compliance_risk",
        "law_section":     "CGST_ACT_2017_S22",
        "risk_description": "GST threshold crossed"
    })
    flags = result["observation"]["compliance_flags"]
    assert len(flags) == 1
    assert "CGST_ACT_2017_S22" in flags


def test_questions_asked_accumulates_in_observation():
    """After ask_clarification(), the observation's questions_asked list grows."""
    sid, obs0 = _reset()
    assert obs0["questions_asked"] == []

    result = _step(sid, {
        "action_type":   "ask_clarification",
        "question_text": "What is the remaining 80C headroom?"
    })
    questions = result["observation"]["questions_asked"]
    assert len(questions) == 1
    assert "80C" in questions[0]


def test_step_count_increments_per_action():
    """step_count in observation must increment exactly once per step() call."""
    sid, obs0 = _reset()
    assert obs0["step_count"] == 0

    r1 = _step(sid, {"action_type": "ask_clarification", "question_text": "Test question here"})
    assert r1["observation"]["step_count"] == 1

    r2 = _step(sid, {"action_type": "ask_clarification", "question_text": "Another question here"})
    assert r2["observation"]["step_count"] == 2


# ── MULTIPLE TASK TYPES ───────────────────────────────────────────────────────

def test_all_three_tasks_reset_successfully():
    """All three task_ids must successfully reset and return a valid observation."""
    for task_id in ["financial_optimize", "tax_planning", "startup_compliance"]:
        sid, obs = _reset(task_id)
        assert obs["task_id"] == task_id, f"task_id mismatch for {task_id}"
        assert obs["done"] == False
        assert len(obs["options"]) > 0


def test_reward_in_valid_range():
    """Every reward returned by /step must be in [-1.0, 1.0]."""
    sid, _ = _reset()
    actions = [
        {"action_type": "ask_clarification", "question_text": "What is the tax regime?"},
        {"action_type": "calculate", "expression": "90000 * 0.3", "expected_result": 27000},
        {"action_type": "choose_option", "option_id": "A"},
    ]
    for action in actions:
        result = _step(sid, action)
        r = result["reward"]
        assert -1.0 <= r <= 1.0, f"Reward {r} out of range for action {action['action_type']}"
        if result["done"]:
            break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
