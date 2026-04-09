# test_graders.py — PROJECT ROOT
# Run: python3 -m pytest test_graders.py -v
# All 20 tests must pass before moving to File 04.
#
# NOTE: All grader outputs are in the open interval (0, 1) — never exactly 0.0 or 1.0.
# _to_open_unit() maps internal [-1, 1] → (0+eps, 1-eps).
# Key mappings:
#   optimal (+1.00) → 0.999999
#   good    (+0.75) → 0.875
#   neutral (+0.40) → 0.70
#   bad     (+0.10) → 0.55
#   harmful (-0.40) → 0.30
#   illegal (-1.00) → 0.000001
#   no-action (0.0) → 0.50

import pytest
from models  import FinancialAction, ProfessionalType
from graders import grade_easy, grade_medium, grade_hard, TIER_REWARD, HARD_CALC_GATE, _to_open_unit


# ── HELPERS ──────────────────────────────────────────────────────────────────

def choose(option: str) -> FinancialAction:
    return FinancialAction(action_type="choose_option", option_id=option)

def calc(expr: str, result: float) -> FinancialAction:
    return FinancialAction(action_type="calculate", expression=expr, expected_result=result)

def flag(section: str) -> FinancialAction:
    return FinancialAction(action_type="flag_compliance_risk", law_section=section)

def escalate(ptype: str) -> FinancialAction:
    return FinancialAction(action_type="escalate_to_professional", professional_type=ProfessionalType(ptype))

def ask() -> FinancialAction:
    return FinancialAction(action_type="ask_clarification", question_text="What is the tax rate?")

# Ground truths for test scenarios
EASY_GT = {
    "outcome_map": {"A": "optimal", "B": "good", "C": "bad", "D": "illegal"},
    "key_calculations": ["90000 * 0.3"],  # = 27000
    "has_compliance_trap": False,
}

MEDIUM_GT = {
    "step_outcome_map": {
        "1": {"A": "optimal", "B": "neutral", "C": "bad"},
        "2": {"A": "optimal", "B": "illegal", "C": "neutral"},
        "3": {"A": "good", "B": "optimal", "C": "bad"},
    },
    "key_calculations": ["14400 + 18000", "224000 * 0"],  # = 32400, 0
    "has_compliance_trap": False,
}

HARD_GT = {
    "outcome_map": {"A": "optimal", "B": "harmful", "C": "bad", "D": "illegal"},
    "key_calculations": ["10000 * (200 - 10)", "1900000 * 0.3"],
    "has_compliance_trap": True,
    "compliance_trap_law_section": "IT_ACT_1961_S17_2_VI",
    "needs_escalation": True,
    "escalation_type": "CA",
}

HARD_GT_NO_TRAP = {
    "outcome_map": {"A": "optimal", "B": "neutral", "C": "bad"},
    "key_calculations": ["500000 * 0.18"],
    "has_compliance_trap": False,
    "needs_escalation": False,
    "escalation_type": "",
}


# ── TIER_REWARD TABLE TESTS ───────────────────────────────────────────────────

def test_tier_reward_all_six_tiers_present():
    """All six OutcomeTier values must be in TIER_REWARD."""
    for tier in ["optimal", "good", "neutral", "bad", "harmful", "illegal"]:
        assert tier in TIER_REWARD, f"Missing tier: {tier}"

def test_tier_reward_illegal_is_minus_one():
    """Illegal MUST be exactly -1.00. This is the unique FLIDIE mechanic."""
    assert TIER_REWARD["illegal"] == -1.00

def test_tier_reward_optimal_is_one():
    assert TIER_REWARD["optimal"] == 1.00

def test_tier_reward_strictly_ordered():
    """optimal > good > neutral > bad > harmful > illegal"""
    order = ["optimal", "good", "neutral", "bad", "harmful", "illegal"]
    vals  = [TIER_REWARD[t] for t in order]
    for i in range(len(vals) - 1):
        assert vals[i] > vals[i+1], f"Order violation: {order[i]} ({vals[i]}) not > {order[i+1]} ({vals[i+1]})"


# ── GRADE_EASY TESTS ─────────────────────────────────────────────────────────

def test_easy_perfect_score():
    """Optimal choice → _to_open_unit(1.0) = 0.999999"""
    score = grade_easy([choose("A")], EASY_GT)
    assert score == _to_open_unit(1.0), f"Optimal choice should score {_to_open_unit(1.0)}, got {score}"
    assert 0.0 < score < 1.0, f"Score must be in open interval (0,1), got {score}"

def test_easy_illegal_choice():
    """Illegal choice → _to_open_unit(-1.0) = 0.000001"""
    score = grade_easy([choose("D")], EASY_GT)
    assert score == _to_open_unit(-1.0), f"Illegal choice should score {_to_open_unit(-1.0)}, got {score}"
    assert 0.0 < score < 1.0, f"Score must be in open interval (0,1), got {score}"

def test_easy_no_action_returns_zero():
    """No choose_option → _to_open_unit(0.0) = 0.5"""
    score = grade_easy([ask(), ask()], EASY_GT)
    assert score == _to_open_unit(0.0), f"No choose_option should return {_to_open_unit(0.0)}, got {score}"
    assert 0.0 < score < 1.0, f"Score must be in open interval (0,1), got {score}"

def test_easy_first_choice_only():
    """Agent chooses B first (good), then A (optimal). Only B counts.
    _to_open_unit(0.75) = 0.875"""
    score = grade_easy([choose("B"), choose("A")], EASY_GT)
    assert score == _to_open_unit(TIER_REWARD["good"]), \
        f"Only first choice counts. Should be {_to_open_unit(TIER_REWARD['good'])}, got {score}"

def test_easy_calculation_bonus():
    """Correct calculate() call earns bonus on top of base score."""
    base  = grade_easy([choose("B")], EASY_GT)
    bonus = grade_easy([calc("90000 * 0.3", 27000), choose("B")], EASY_GT)
    assert bonus > base, f"Correct calculation should add bonus. base={base}, with_calc={bonus}"


# ── GRADE_MEDIUM TESTS ───────────────────────────────────────────────────────

def test_medium_all_optimal_steps():
    """All 3 steps optimal → high score (+ possible calc bonus)."""
    actions = [choose("A"), choose("A"), choose("B")]
    score = grade_medium(actions, MEDIUM_GT)
    assert score >= 0.90, f"All optimal steps should score ≥ 0.90, got {score}"

def test_medium_illegal_step_tanks_score():
    """Step 2 illegal choice tanks the weighted average → below 0.50"""
    actions = [choose("A"), choose("B"), choose("B")]  # step 2 B = illegal
    score = grade_medium(actions, MEDIUM_GT)
    assert score < 0.70, f"Illegal step 2 should drop score below 0.70, got {score}"

def test_medium_calculation_bonus():
    """Two correct calculations earn bonus → with_calc > no_calc"""
    no_calc   = grade_medium([choose("B"), choose("A"), choose("A")], MEDIUM_GT)
    with_calc = grade_medium([
        calc("14400 + 18000", 32400), calc("224000 * 0", 0),
        choose("B"), choose("A"), choose("A")
    ], MEDIUM_GT)
    assert with_calc > no_calc, f"Calculations should add bonus. no_calc={no_calc}, with={with_calc}"


# ── GRADE_HARD TESTS ─────────────────────────────────────────────────────────

def test_hard_perfect_score_all_dimensions():
    """Optimal choice + correct trap + correct escalation + correct calc → ≥ 0.85"""
    actions = [
        flag("IT_ACT_1961_S17_2_VI"),
        escalate("CA"),
        calc("10000 * (200 - 10)", 1900000),
        calc("1900000 * 0.3", 570000),
        choose("A"),
    ]
    score = grade_hard(actions, HARD_GT)
    assert score >= 0.85, f"Perfect hard case should score ≥ 0.85, got {score}"

def test_hard_illegal_choice_no_calc_bonus():
    """Illegal choice → gate fails → calc bonus blocked. Both scores identical."""
    actions_no_calc   = [choose("D")]
    actions_with_calc = [
        calc("10000 * (200 - 10)", 1900000),
        calc("1900000 * 0.3", 570000),
        choose("D"),
    ]
    score_no_calc   = grade_hard(actions_no_calc,   HARD_GT)
    score_with_calc = grade_hard(actions_with_calc, HARD_GT)
    assert score_no_calc == score_with_calc, (
        f"Illegal choice should block calc bonus. "
        f"no_calc={score_no_calc}, with_calc={score_with_calc}"
    )

def test_hard_false_alarm_penalty():
    """Flag compliance when no trap exists → score drops."""
    with_flag    = grade_hard([flag("IT_ACT_1961_S17_2_VI"), choose("A")], HARD_GT_NO_TRAP)
    without_flag = grade_hard([choose("A")],                                HARD_GT_NO_TRAP)
    assert with_flag < without_flag, \
        f"False alarm should lower score. with={with_flag}, without={without_flag}"

def test_hard_wrong_trap_section_no_credit():
    """Flag the wrong law section → same score as not flagging."""
    correct_flag = grade_hard([flag("IT_ACT_1961_S17_2_VI"), choose("A")], HARD_GT)
    wrong_flag   = grade_hard([flag("CGST_ACT_2017_S22"),    choose("A")], HARD_GT)
    no_flag      = grade_hard([choose("A")],                               HARD_GT)
    assert correct_flag > wrong_flag, f"Correct flag should outscore wrong flag"
    assert wrong_flag == no_flag,     f"Wrong flag section = same as not flagging"

def test_hard_unnecessary_escalation_penalised():
    """Escalate when needs_escalation=False → score drops."""
    with_esc    = grade_hard([escalate("CA"), choose("A")], HARD_GT_NO_TRAP)
    without_esc = grade_hard([choose("A")],                 HARD_GT_NO_TRAP)
    assert with_esc < without_esc, \
        f"Unnecessary escalation should lower score. with={with_esc}, without={without_esc}"

def test_hard_calc_gate_threshold_value():
    """HARD_CALC_GATE must equal 0.40 × 0.30 = 0.12."""
    assert abs(HARD_CALC_GATE - 0.12) < 0.001, \
        f"HARD_CALC_GATE should be 0.12, got {HARD_CALC_GATE}"

def test_hard_empty_history_returns_zero():
    """Empty history → _to_open_unit(0.0) = 0.5"""
    score = grade_hard([], HARD_GT)
    assert score == _to_open_unit(0.0), \
        f"Empty history should return {_to_open_unit(0.0)}, got {score}"
    assert 0.0 < score < 1.0, f"Score must be in open interval (0,1), got {score}"


# ── DETERMINISM TESTS ────────────────────────────────────────────────────────

def test_all_graders_are_deterministic():
    """Run each grader 100 times. All scores must be identical."""
    actions = [flag("IT_ACT_1961_S17_2_VI"), escalate("CA"), calc("90000 * 0.3", 27000), choose("A")]

    easy_scores   = {grade_easy([choose("A")], EASY_GT) for _ in range(100)}
    assert len(easy_scores)   == 1, f"grade_easy is non-deterministic: {easy_scores}"

    medium_scores = {grade_medium([choose("A"), choose("A"), choose("B")], MEDIUM_GT) for _ in range(100)}
    assert len(medium_scores) == 1, f"grade_medium is non-deterministic: {medium_scores}"

    hard_scores   = {grade_hard(actions, HARD_GT) for _ in range(100)}
    assert len(hard_scores)   == 1, f"grade_hard is non-deterministic: {hard_scores}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])