# graders.py — PROJECT ROOT
# ─────────────────────────────────────────────────────────────────────────────
# FLIDIE Grader Functions
# Finance & Legal Decision Intelligence Environment
#
# GRADER CONTRACT (all three functions must obey):
#   Input : List[FinancialAction], ground_truth dict, **kwargs
#   Output: float in (0.1, 0.99) inclusive
#   Side effects: NONE. No mutations, no I/O, no randomness, no LLM calls.
#   Determinism: same inputs → same output. Always. Without exception.
# ─────────────────────────────────────────────────────────────────────────────

from typing import List, Dict, Any, Optional
from models import FinancialAction, OutcomeTier, ProfessionalType


# ── OUTPUT NORMALISATION ─────────────────────────────────────────────────────

_EPS = 0.02


def _to_open_unit(raw: float = 0.0) -> float:
    normalised = (raw + 1) / 2.0
    clamped    = max(_EPS, min(1 - _EPS, normalised))
    return round(clamped, 4)


# ── MASTER REWARD TABLE ──────────────────────────────────────────────────────

TIER_REWARD: Dict[str, float] = {
    "optimal": 0.95,
    "good":    0.75,
    "neutral": 0.40,
    "bad":     0.10,
    "harmful": -0.40,
    "illegal": -0.95,
}

MEDIUM_STEP_WEIGHTS = {1: 0.40, 2: 0.35, 3: 0.25}
CALC_TOLERANCE = 0.01

HARD_WEIGHTS = {
    "decision":    0.40,
    "trap":        0.25,
    "escalation":  0.20,
    "calculation": 0.15,
}

HARD_CALC_GATE = 0.12


# ── HELPERS ──────────────────────────────────────────────────────────────────

def _safe_eval(expression: str, expected: Optional[float] = None) -> Optional[float]:
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return float(result)
    except Exception:
        return None


def _calc_bonus(
    action_history: List[FinancialAction] = None,
    key_calculations: List[str] = None,
    max_bonus: float = 0.10,
) -> float:
    action_history   = action_history   or []
    key_calculations = key_calculations or []

    if not key_calculations:
        return 0.0

    gt_results = []
    for expr in key_calculations:
        val = _safe_eval(expr)
        if val is not None:
            gt_results.append(val)

    if not gt_results:
        return 0.0

    calc_actions = []
    seen_exprs = set()
    for action in action_history:
        if (action.action_type == "calculate"
                and action.expression
                and action.expression not in seen_exprs):
            calc_actions.append(action)
            seen_exprs.add(action.expression)

    if not calc_actions:
        return 0.0

    matched = 0
    for gt_val in gt_results:
        for action in calc_actions:
            agent_val = _safe_eval(action.expression, action.expected_result)
            if agent_val is not None:
                tolerance = abs(gt_val) * CALC_TOLERANCE if gt_val != 0 else 0.01
                if abs(agent_val - gt_val) <= tolerance:
                    matched += 1
                    break

    fraction = matched / len(gt_results)
    return round(fraction * max_bonus, 4)


# ── GRADE_EASY ───────────────────────────────────────────────────────────────

def grade_easy(
    action_history: List[FinancialAction] = None,
    ground_truth:   Dict[str, Any]        = None,
    **kwargs,
) -> float:
    action_history = action_history or []
    ground_truth   = ground_truth   or {}

    choose_action = next(
        (a for a in action_history if a.action_type == "choose_option"),
        None
    )

    if choose_action is None or choose_action.option_id is None:
        return _to_open_unit(0.1)

    outcome_map = ground_truth.get("outcome_map", {})
    tier        = outcome_map.get(choose_action.option_id)

    if tier is None:
        return _to_open_unit(0.1)

    base_score = TIER_REWARD.get(tier, 0.1)
    key_calcs  = ground_truth.get("key_calculations", [])
    calc_bonus = _calc_bonus(action_history, key_calcs, max_bonus=0.10)

    return _to_open_unit(base_score + calc_bonus)


# ── GRADE_MEDIUM ─────────────────────────────────────────────────────────────

def grade_medium(
    action_history: List[FinancialAction]        = None,
    ground_truth:   Dict[str, Any]               = None,
    step_log:       Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> float:
    action_history = action_history or []
    ground_truth   = ground_truth   or {}

    step_outcome_map = ground_truth.get("step_outcome_map", {})
    key_calcs        = ground_truth.get("key_calculations", [])

    choose_actions = [
        a for a in action_history
        if a.action_type == "choose_option" and a.option_id
    ]

    if not choose_actions:
        return _to_open_unit(0.1)

    if step_log:
        step_decisions = {
            entry["step_num"]: entry["option_id"]
            for entry in step_log
            if "step_num" in entry and "option_id" in entry
        }
    else:
        step_decisions = {
            i + 1: action.option_id
            for i, action in enumerate(choose_actions)
        }

    total_weight_used = 0.0
    weighted_score    = 0.0

    for step_num, option_id in step_decisions.items():
        step_str = str(step_num)
        step_map = step_outcome_map.get(step_str, {})
        tier     = step_map.get(option_id)
        weight   = MEDIUM_STEP_WEIGHTS.get(step_num, 0.25)

        if tier is not None:
            step_score        = TIER_REWARD.get(tier, 0.1)
            weighted_score   += step_score * weight
            total_weight_used += weight

    if total_weight_used > 0 and total_weight_used < 1:
        weighted_score = weighted_score / total_weight_used

    calc_bonus = _calc_bonus(action_history, key_calcs, max_bonus=0.20)

    return _to_open_unit(weighted_score + calc_bonus)


# ── GRADE_HARD ───────────────────────────────────────────────────────────────

def grade_hard(
    action_history: List[FinancialAction] = None,
    ground_truth:   Dict[str, Any]        = None,
    **kwargs,
) -> float:
    action_history = action_history or []
    ground_truth   = ground_truth   or {}

    outcome_map  = ground_truth.get("outcome_map",                {})
    key_calcs    = ground_truth.get("key_calculations",            [])
    has_trap     = ground_truth.get("has_compliance_trap",         False)
    trap_section = ground_truth.get("compliance_trap_law_section", "")
    needs_esc    = ground_truth.get("needs_escalation",            False)
    esc_type     = ground_truth.get("escalation_type",             "")

    choose_action = next(
        (a for a in action_history if a.action_type == "choose_option"),
        None
    )
    decision_raw = 0.1
    if choose_action and choose_action.option_id:
        tier         = outcome_map.get(choose_action.option_id)
        decision_raw = TIER_REWARD.get(tier, 0.1) if tier else 0.1

    score = HARD_WEIGHTS["decision"] * decision_raw

    flag_actions = [
        a for a in action_history
        if a.action_type == "flag_compliance_risk"
    ]

    trap_component = 0.1
    if flag_actions:
        if has_trap:
            correctly_flagged = any(
                a.law_section == trap_section
                for a in flag_actions
                if a.law_section
            )
            trap_component = 0.99 if correctly_flagged else 0.1
        else:
            trap_component = -0.40
    else:
        trap_component = 0.1

    score += HARD_WEIGHTS["trap"] * trap_component

    esc_actions  = [a for a in action_history if a.action_type == "escalate_to_professional"]
    did_escalate = len(esc_actions) > 0

    escalation_component = 0.1
    if needs_esc:
        if did_escalate:
            correct_type_match = any(
                a.professional_type
                and a.professional_type.value.upper() == esc_type.upper()
                for a in esc_actions
            )
            escalation_component = 0.99 if correct_type_match else 0.15
        else:
            escalation_component = 0.1
    else:
        escalation_component = -0.40 if did_escalate else 0.1

    score += HARD_WEIGHTS["escalation"] * escalation_component

    if decision_raw >= HARD_CALC_GATE:
        calc_bonus = _calc_bonus(action_history, key_calcs, max_bonus=1)
        score += HARD_WEIGHTS["calculation"] * calc_bonus

    return _to_open_unit(score)
