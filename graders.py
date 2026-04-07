# graders.py — PROJECT ROOT
# ─────────────────────────────────────────────────────────────────────────────
# FLIDIE Grader Functions
# Finance & Legal Decision Intelligence Environment
#
# GRADER CONTRACT (all three functions must obey):
#   Input : List[FinancialAction], ground_truth dict, **kwargs
#   Output: float in [−1.0, 1.0]
#   Side effects: NONE. No mutations, no I/O, no randomness, no LLM calls.
#   Determinism: same inputs → same output. Always. Without exception.
#
# These are pure functions. They can be tested in complete isolation from the
# environment, the server, and the inference script. This is intentional.
# ─────────────────────────────────────────────────────────────────────────────

from typing import List, Dict, Any, Optional
from models import FinancialAction, OutcomeTier, ProfessionalType


# ── MASTER REWARD TABLE ──────────────────────────────────────────────────────
# Single source of truth. Change here, changes everywhere.
# Do NOT hardcode 1.0 or -0.40 anywhere in grader logic — always use TIER_REWARD.

TIER_REWARD: Dict[str, float] = {
    "optimal": 1.00,
    "good":    0.75,
    "neutral": 0.40,
    "bad":     0.10,
    "harmful": -0.40,
    "illegal": -1.00,
}

# Step weights for medium grader (must sum to 1.0)
MEDIUM_STEP_WEIGHTS = {1: 0.40, 2: 0.35, 3: 0.25}

# Calculation tolerance — how close does expected_result need to be?
# 1% tolerance handles floating-point rounding in GST/tax calculations.
CALC_TOLERANCE = 0.01

# Hard grader dimension weights (must sum to 1.0)
HARD_WEIGHTS = {
    "decision":    0.40,
    "trap":        0.25,
    "escalation":  0.20,
    "calculation": 0.15,
}

# Gate threshold: hard calculation bonus only applies if decision_raw >= this
# 0.12 = 0.40 (decision weight) × 0.30 (equivalent to "bad" tier partially)
# Effectively: if you gave illegal advice, you get no calculation bonus.
HARD_CALC_GATE = 0.12


# ── HELPER: Safe eval for calculate() actions ────────────────────────────────

def _safe_eval(expression: str, expected: Optional[float]) -> Optional[float]:
    """
    Evaluate an arithmetic expression in a sandboxed namespace.
    Returns the numeric result if valid, None if expression fails.

    Security: uses a restricted namespace with no builtins.
    Only arithmetic operators and float literals are permitted.
    This prevents code injection through the calculate() action.

    Args:
        expression: Python arithmetic string e.g. "1900000 * 0.3"
        expected:   Agent's stated expected_result (for logging, not used here)

    Returns:
        float if expression is valid arithmetic, None otherwise
    """
    try:
        # Restricted namespace: no builtins, no imports, pure arithmetic
        result = eval(expression, {"__builtins__": {}}, {})
        return float(result)
    except Exception:
        return None


def _calc_bonus(
    action_history: List[FinancialAction],
    key_calculations: List[str],
    max_bonus: float = 0.10,
) -> float:
    """
    Score all calculate() actions in the episode against the key_calculations
    list from ground_truth.

    Matching logic:
    1. Evaluate each key_calculation string (ground truth arithmetic)
    2. For each calculate() action in history, evaluate its expression
    3. If agent's result is within CALC_TOLERANCE of ground truth result: match
    4. Score = (matched / total_key_calcs) × max_bonus

    If key_calculations is empty, returns 0.0 (no bonus available).
    First calculate() action per unique expression is counted; duplicates ignored.
    """
    if not key_calculations:
        return 0.0

    # Evaluate ground truth answers
    gt_results = []
    for expr in key_calculations:
        val = _safe_eval(expr, None)
        if val is not None:
            gt_results.append(val)

    if not gt_results:
        return 0.0

    # Collect unique calculate() actions (deduplicate by expression)
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

    # Count how many ground truth calculations were correctly verified
    matched = 0
    for gt_val in gt_results:
        for action in calc_actions:
            agent_val = _safe_eval(action.expression, action.expected_result)
            if agent_val is not None:
                tolerance = abs(gt_val) * CALC_TOLERANCE if gt_val != 0 else 0.01
                if abs(agent_val - gt_val) <= tolerance:
                    matched += 1
                    break  # Each ground truth calc matched at most once

    fraction = matched / len(gt_results)
    return round(fraction * max_bonus, 4)


# ── GRADE_EASY ───────────────────────────────────────────────────────────────

def grade_easy(
    action_history: List[FinancialAction],
    ground_truth:   Dict[str, Any],
) -> float:
    """
    Grade a single-step financial decision scenario.

    Scoring:
        base_score  = TIER_REWARD[outcome_map[option_id]]
        calc_bonus  = _calc_bonus(actions, key_calculations, max_bonus=0.10)
        final       = clamp(base_score + calc_bonus, -1.0, 1.0)

    Only the FIRST choose_option action is scored. If the agent
    chooses twice (e.g. second-guesses itself), only the first counts.
    This prevents agents from trying all four options and keeping the best.

    Edge cases:
        No choose_option action → return 0.0 (episode timed out / agent confused)
        option_id not in outcome_map → return 0.0 (defensive fallback)
        outcome tier not in TIER_REWARD → return 0.0 (bad data fallback)
    """
    # Find the FIRST choose_option action
    choose_action = next(
        (a for a in action_history if a.action_type == "choose_option"),
        None
    )

    if choose_action is None or choose_action.option_id is None:
        return 0.0

    outcome_map = ground_truth.get("outcome_map", {})
    tier = outcome_map.get(choose_action.option_id)

    if tier is None:
        return 0.0  # option_id not in map — defensive fallback

    base_score = TIER_REWARD.get(tier, 0.0)

    # Calculation quality bonus (max +0.10 for easy)
    key_calcs = ground_truth.get("key_calculations", [])
    calc_bonus = _calc_bonus(action_history, key_calcs, max_bonus=0.10)

    return round(max(-1.0, min(1.0, base_score + calc_bonus)), 4)


# ── GRADE_MEDIUM ─────────────────────────────────────────────────────────────

def grade_medium(
    action_history: List[FinancialAction],
    ground_truth:   Dict[str, Any],
    step_log:       Optional[List[Dict[str, Any]]] = None,
) -> float:
    """
    Grade a multi-step tax planning scenario.

    Medium scenarios have up to 3 sequential decision points. Each step has
    its own choose_option action and its own outcome map in step_outcome_map.
    The final score is a weighted average across steps, plus a calculation bonus.

    Step weight allocation (must sum to 1.0):
        Step 1: 40%  — usually the regime / strategy choice (most impactful)
        Step 2: 35%  — usually the detailed calculation decision
        Step 3: 25%  — usually the compliance/filing decision

    Args:
        action_history: All actions taken during episode.
        ground_truth:   Dict with step_outcome_map and key_calculations.
        step_log:       Optional list of {step_num, option_id} dicts from
                        environment state. If None, infers step assignments
                        from action order (less accurate but functional).

    Scoring:
        weighted_score = Σ(TIER_REWARD[step_tier] × step_weight)
        calc_bonus     = _calc_bonus(actions, key_calculations, max_bonus=0.20)
        final          = clamp(weighted_score + calc_bonus, -1.0, 1.0)
    """
    step_outcome_map = ground_truth.get("step_outcome_map", {})
    key_calcs        = ground_truth.get("key_calculations", [])

    # Collect all choose_option actions in order
    choose_actions = [
        a for a in action_history
        if a.action_type == "choose_option" and a.option_id
    ]

    if not choose_actions:
        return 0.0

    # If step_log is provided, use it for precise step assignments.
    # Otherwise, assign steps sequentially: 1st choose→step1, 2nd→step2, etc.
    if step_log:
        step_decisions = {
            entry["step_num"]: entry["option_id"]
            for entry in step_log
            if "step_num" in entry and "option_id" in entry
        }
    else:
        # Sequential assignment fallback
        step_decisions = {
            i + 1: action.option_id
            for i, action in enumerate(choose_actions)
        }

    # Compute weighted score across all steps
    total_weight_used = 0.0
    weighted_score    = 0.0

    for step_num, option_id in step_decisions.items():
        step_str    = str(step_num)
        step_map    = step_outcome_map.get(step_str, {})
        tier        = step_map.get(option_id)
        weight      = MEDIUM_STEP_WEIGHTS.get(step_num, 0.25)

        if tier is not None:
            step_score      = TIER_REWARD.get(tier, 0.0)
            weighted_score += step_score * weight
            total_weight_used += weight

    # Normalize if agent didn't reach all steps (e.g. closed early)
    if total_weight_used > 0 and total_weight_used < 1.0:
        weighted_score = weighted_score / total_weight_used

    # Calculation bonus (max +0.20 for medium — bigger reward for doing the math)
    calc_bonus = _calc_bonus(action_history, key_calcs, max_bonus=0.20)

    return round(max(-1.0, min(1.0, weighted_score + calc_bonus)), 4)


# ── GRADE_HARD ───────────────────────────────────────────────────────────────

def grade_hard(
    action_history: List[FinancialAction],
    ground_truth:   Dict[str, Any],
) -> float:
    """
    Grade a startup compliance audit scenario across 4 independent dimensions.

    Dimensions and weights (sum to 1.0):
        decision     (0.40): Was the final advice correct?
        trap         (0.25): Did the agent identify the compliance trap?
        escalation   (0.20): Did the agent correctly handle professional referral?
        calculation  (0.15): Did the agent verify the key arithmetic? [GATED]

    GATE on calculation dimension:
        The calculation bonus only applies if decision_raw >= HARD_CALC_GATE (0.12).
        This prevents agents from collecting calculation points while giving
        catastrophically wrong advice. The gate value 0.12 = 0.40 × 0.30,
        which means any agent that didn't choose the "illegal" option passes.

    Trap Detection logic (has_compliance_trap=true):
        +1.00 if agent called flag_compliance_risk() with exact compliance_trap_law_section
         0.00 if trap exists but agent never flagged it (no penalty for missing)
        −0.40 if trap does NOT exist but agent flagged anyway (false alarm penalty)

    Escalation logic:
        +1.00 if needs_escalation=true AND escalation_type matches
        +0.15 if needs_escalation=true AND wrong professional type
         0.00 if needs_escalation=true AND agent never escalated
        −0.40 if needs_escalation=false AND agent escalated
        +1.00 if needs_escalation=false AND agent did NOT escalate
    """
    outcome_map  = ground_truth.get("outcome_map",               {})
    key_calcs    = ground_truth.get("key_calculations",           [])
    has_trap     = ground_truth.get("has_compliance_trap",        False)
    trap_section = ground_truth.get("compliance_trap_law_section", "")
    needs_esc    = ground_truth.get("needs_escalation",            False)
    esc_type     = ground_truth.get("escalation_type",             "")

    # ── DIMENSION 1: DECISION (weight 0.40) ──────────────────────────────────
    # First choose_option action only. Same rule as grade_easy.
    choose_action    = next(
        (a for a in action_history if a.action_type == "choose_option"),
        None
    )
    decision_raw = 0.0
    if choose_action and choose_action.option_id:
        tier         = outcome_map.get(choose_action.option_id)
        decision_raw = TIER_REWARD.get(tier, 0.0) if tier else 0.0

    score = HARD_WEIGHTS["decision"] * decision_raw

    # ── DIMENSION 2: TRAP DETECTION (weight 0.25) ────────────────────────────
    # Find all flag_compliance_risk actions in the episode
    flag_actions = [
        a for a in action_history
        if a.action_type == "flag_compliance_risk"
    ]

    trap_component = 0.0
    if flag_actions:
        if has_trap:
            # Check if ANY flag action cited the exact correct law section
            correctly_flagged = any(
                a.law_section == trap_section
                for a in flag_actions
                if a.law_section
            )
            trap_component = 1.0 if correctly_flagged else 0.0
            # Wrong section: 0.0 — not penalized for trying, just no credit
        else:
            # No trap in this scenario — flagging is a false alarm
            trap_component = -0.40
    else:
        # No flag actions at all
        if has_trap:
            trap_component = 0.0   # Missed the trap — no bonus, no penalty
        else:
            trap_component = 0.0   # Correctly didn't flag when no trap
            # Note: this means not-flagging on a non-trap scenario earns 0.0
            # (not +1.0). The hard grader doesn't reward "doing nothing".
            # If you want to reward correct non-flagging, change this to +1.0
            # and adjust the weight. Current design: trap detection is optional.

    score += HARD_WEIGHTS["trap"] * trap_component

    # ── DIMENSION 3: ESCALATION (weight 0.20) ────────────────────────────────
    esc_actions = [
        a for a in action_history
        if a.action_type == "escalate_to_professional"
    ]
    did_escalate = len(esc_actions) > 0

    escalation_component = 0.0
    if needs_esc:
        if did_escalate:
            # Check professional type match
            correct_type_match = any(
                a.professional_type
                and a.professional_type.value.upper() == esc_type.upper()
                for a in esc_actions
            )
            escalation_component = 1.0 if correct_type_match else 0.15
        else:
            escalation_component = 0.0   # Needed but didn't escalate
    else:
        if did_escalate:
            escalation_component = -0.40  # Unnecessary escalation
        else:
            escalation_component = 0.0   # Correctly didn't escalate (no bonus)

    score += HARD_WEIGHTS["escalation"] * escalation_component

    # ── DIMENSION 4: CALCULATION QUALITY (weight 0.15, GATED) ────────────────
    # Gate: only apply if decision_raw >= HARD_CALC_GATE
    # This ensures calculation bonus doesn't flow to agents that gave illegal advice
    if decision_raw >= HARD_CALC_GATE:
        calc_bonus = _calc_bonus(action_history, key_calcs, max_bonus=1.0)
        score += HARD_WEIGHTS["calculation"] * calc_bonus

    return round(max(-1.0, min(1.0, score)), 4)
