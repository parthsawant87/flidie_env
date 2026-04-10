# flidie_environment.py — PROJECT ROOT
# ─────────────────────────────────────────────────────────────────────────────
# FlidieEnv — FLIDIE Core Environment Class
# Finance & Legal Decision Intelligence Environment
#
# Design guarantees:
#   1. State is owned here and nowhere else. No other file modifies _state.
#   2. Ground truth NEVER appears in any Observation returned by this class.
#   3. All randomness is isolated to reset(). step() is deterministic after reset().
#   4. Graders are pure functions called once per episode. Not called on each step.
#   5. Zero LLM calls. Zero external API calls. Zero randomness in step().
#   6. Thread safety: NOT thread-safe. Each HTTP session gets its own instance.
# ─────────────────────────────────────────────────────────────────────────────

import json
import random
from pathlib import Path
from typing  import List, Optional, Dict, Any

from models  import (
    FinancialAction, ScenarioObservation, StepResult,
    FinancialSnapshot, FinancialOption, FinancialCategory
)
from graders import grade_easy, grade_medium, grade_hard, _to_open_unit
from tasks   import get_task, list_tasks, TaskConfig


class FlidieEnv:
    """
    FLIDIE Core Environment — OpenEnv compliant.

    An AI agent interacts with this class through three public methods:
        reset(task_id)  → ScenarioObservation
        step(action)    → StepResult
        state()         → dict

    The agent faces Indian financial and legal scenarios and must choose
    the optimal action from four options using FinancialAction.

    Thread safety: NOT thread-safe. Each HTTP session gets its own
    FlidieEnv instance via the session registry in server/app.py.
    Never share one FlidieEnv instance across concurrent requests.
    """

    # Timeout penalty — stronger than clinical triage (-0.2) because
    # in finance, failing to advise leaves the client in the wrong state.
    TIMEOUT_PENALTY: float = 0.0001

    # Redundant action penalty — universal for all action types
    REDUNDANT_PENALTY: float = -0.05

    def __init__(self) -> None:
        # All mutable state lives in exactly these fields — nowhere else.
        self._state:            Dict[str, Any]       = {}
        self._action_history:   List[FinancialAction] = []
        self._step_count:       int                  = 0
        self._current_scenario: Optional[dict]       = None
        self._current_task:     Optional[TaskConfig] = None
        self._done:             bool                 = False
        self._data_cache:       Dict[str, list]      = {}  # JSON read-once cache
        # step_log tracks which option was chosen at which step
        # Used by grade_medium() for accurate step assignment
        self._step_log:         List[Dict[str, Any]] = []

    # ── DATA LOADING ─────────────────────────────────────────────────────────

    def _load_scenarios(self, filename: str) -> list:
        """
        Load scenario data from JSON file with in-memory caching.
        First call reads from disk. All subsequent calls return the cached list.
        Prevents re-reading JSON on every reset() during batch evaluation.
        """
        if filename not in self._data_cache:
            data_path = Path(__file__).parent / filename
            if not data_path.exists():
                raise FileNotFoundError(
                    f"Scenario file not found: {data_path}\n"
                    f"Ensure {filename} exists at project root."
                )
            with open(data_path, "r", encoding="utf-8") as f:
                scenarios = json.load(f)
            if not scenarios:
                raise ValueError(f"Scenario file {filename} is empty. Add at least 10 scenarios.")
            self._data_cache[filename] = scenarios
        return self._data_cache[filename]

    # ── RESET ────────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "financial_optimize") -> ScenarioObservation:
        """
        Start a new episode. Returns the initial observation.

        Sequence:
        1. Validate and load task configuration.
        2. Randomly select a scenario from the task's data file.
        3. Reset ALL mutable state to clean values.
        4. Construct and return initial ScenarioObservation.

        PRIVACY BOUNDARY: ground_truth fields are stored in
        self._current_scenario["ground_truth"] but are NEVER included
        in the ScenarioObservation. The agent must never see the answer.

        Args:
            task_id: One of "financial_optimize", "tax_planning",
                     "startup_compliance". Defaults to easy task.

        Returns:
            ScenarioObservation: The initial observation for the new episode.

        Raises:
            ValueError: If task_id is not registered in TASK_REGISTRY.
            FileNotFoundError: If task's data file does not exist.
        """
        # Step 1: Load task config — raises ValueError for unknown task_id
        self._current_task = get_task(task_id)

        # Step 2: Load scenarios and pick one at random.
        # random.choice is the ONLY source of randomness. step() is deterministic after this.
        scenarios = self._load_scenarios(self._current_task.data_file)
        self._current_scenario = random.choice(scenarios)

        # Step 3: Reset ALL mutable state.
        self._action_history = []
        self._step_count     = 0
        self._step_log       = []
        self._done           = False

        # Step 4: Build initial observation.
        # ONLY include fields the agent should see. Ground truth stays hidden.
        obs = self._build_observation(step_number=1)
        self._state = obs.model_dump()
        return obs

    # ── STEP ─────────────────────────────────────────────────────────────────

    def step(self, action: FinancialAction) -> StepResult:
        """
        Process one agent action and return the result.

        Sequence:
        1. Guard: check episode is initialized and not already done.
        2. Detect redundant consecutive action (same type as previous).
        3. Record action and increment step counter.
        4. Apply action side-effects (update observation state).
        5. Check termination: choose_option or timeout.
        6. Compute reward: final (grader) or intermediate.
        7. Return StepResult.

        Args:
            action: A validated FinancialAction Pydantic model.

        Returns:
            StepResult: New observation, reward, done flag, info dict.

        Raises:
            RuntimeError: If called on terminal or uninitialized episode.
        """
        # Guard 1: No stepping on finished episodes.
        if self._done:
            raise RuntimeError(
                "Episode is over. Call reset() before calling step() again."
            )

        # Guard 2: No stepping before reset().
        if self._current_task is None:
            raise RuntimeError(
                "Environment not initialized. Call reset(task_id) first."
            )

        # Redundant action detection: same action_type as the previous step.
        is_redundant = (
            len(self._action_history) > 0
            and self._action_history[-1].action_type == action.action_type
        )

        # Record action and update step counter.
        self._action_history.append(action)
        self._step_count += 1

        # Apply side-effects: update state based on action type.
        self._apply_side_effects(action)

        # Log the step for grade_medium() step assignment.
        if action.action_type == "choose_option" and action.option_id:
            current_step_num = len([
                e for e in self._step_log
                if e.get("type") == "choose_option"
            ]) + 1
            self._step_log.append({
                "step_num":  current_step_num,
                "option_id": action.option_id,
                "type":       "choose_option",
            })

        # Check termination conditions.
        done   = False
        reward = 0.001

        if action.action_type == "choose_option":
            # For multi-step scenarios: only terminal on the LAST step.
            # Check if there are more steps to go.
            more_steps = self._has_more_steps()
            if not more_steps:
                # Final decision reached — compute terminal reward.
                done   = True
                reward = self._compute_final_reward()
            else:
                # More steps remain in multi-step scenario.
                # Advance to next step's prompt and give small intermediate reward.
                reward = 0.001  # Neutral for advancing — graded at end
                self._advance_step()

        elif self._step_count >= self._current_task.max_steps:
            # Timeout: agent used all steps without completing the scenario.
            done   = True
            reward = self.TIMEOUT_PENALTY

        else:
            # Non-terminal, non-timeout: compute intermediate reward.
            reward = (
                self.REDUNDANT_PENALTY
                if is_redundant
                else self._intermediate_reward(action)
            )

        # Update done flag in state.
        if done:
            self._done          = True
            self._state["done"] = True

        # Ensure reward is strictly inside (0, 1) for every step.
        # Final rewards from graders already went through _to_open_unit().
        # Intermediate rewards are raw [-1, 1] nudges — normalise them too.
        _EPS = 1e-4
        if done:
            reward_clamped = max(_EPS, min(1 - _EPS, reward))
        else:
            reward_clamped = _to_open_unit(reward)

        # Rebuild observation from current state.
        self._state["step_count"] = self._step_count
        obs = ScenarioObservation.model_validate(self._state)

        return StepResult(
            observation = obs,
            reward      = round(reward_clamped, 4),
            done        = done,
            info        = {
                "step":         self._step_count,
                "task_id":      self._current_task.task_id,
                "max_steps":    self._current_task.max_steps,
                "is_redundant": is_redundant,
                "scenario_id":  self._current_scenario["id"],
                "action_taken": action.action_type,
            }
        )

    # ── MULTI-STEP HELPERS ───────────────────────────────────────────────────

    def _has_more_steps(self) -> bool:
        """
        Returns True if the current scenario has more decision steps remaining.
        Easy scenarios have no step_sequence — always False (single-step).
        Medium/hard have step_sequence — check if all steps have been decided.
        """
        step_sequence = self._current_scenario.get("step_sequence", [])
        if not step_sequence:
            return False  # Easy: no sequence
        choices_made = len([e for e in self._step_log if e.get("type") == "choose_option"])
        return choices_made < len(step_sequence)

    def _advance_step(self) -> None:
        """
        Move to the next step in a multi-step scenario.
        Updates current_step_context and current_step_number in state.
        Also updates the options to show the current step's choices.
        """
        choices_made = len([e for e in self._step_log if e.get("type") == "choose_option"])
        step_sequence = self._current_scenario.get("step_sequence", [])

        if choices_made < len(step_sequence):
            next_step = step_sequence[choices_made]
            self._state["current_step_number"] = next_step["step"]
            self._state["current_step_context"] = next_step.get("prompt", "")
            context_update = next_step.get("context_update", "")
            if context_update:
                self._state["current_step_context"] += " | " + context_update
            # Update options for the next step
            raw_opts = next_step.get("options", [])
            self._state["options"] = [
                FinancialOption(**opt).model_dump() for opt in raw_opts
            ]

    # ── SIDE EFFECTS ────────────────────────────────────────────────────────

    def _apply_side_effects(self, action: FinancialAction) -> None:
        """
        Update observation state based on non-terminal action side-effects.
        Does NOT affect grading — purely informational state updates.
        """
        if action.action_type == "ask_clarification" and action.question_text:
            questions = self._state.get("questions_asked", [])
            questions.append(action.question_text[:200])  # Truncate long questions
            self._state["questions_asked"] = questions

        elif action.action_type == "calculate" and action.expression:
            calcs = self._state.get("calculations_done", [])
            # Evaluate and store result for agent to see
            try:
                result = eval(action.expression, {"__builtins__": {}}, {})
                calcs.append({
                    "expression": action.expression,
                    "result":     round(float(result), 4),
                    "expected":   action.expected_result,
                })
            except Exception:
                calcs.append({
                    "expression": action.expression,
                    "result":     None,
                    "expected":   action.expected_result,
                    "error":      "invalid_expression",
                })
            self._state["calculations_done"] = calcs

        elif action.action_type == "flag_compliance_risk" and action.law_section:
            flags = self._state.get("compliance_flags", [])
            flags.append(action.law_section)
            self._state["compliance_flags"] = flags

    # ── INTERMEDIATE REWARD ─────────────────────────────────────────────────

    def _intermediate_reward(self, action: FinancialAction) -> float:
        """
        Per-step reward for non-terminal, non-redundant actions.
        Uses the reward table from File 03 § 04.
        Reads ground_truth to compute context-dependent rewards.
        Returns float in [-0.10, +0.10] — nudges, doesn't dominate.
        """
        gt = self._current_scenario.get("ground_truth", {})

        if action.action_type == "calculate":
            if not action.expression:
                return 0.001
            # Check if same expression was already submitted (redundant calculate)
            prev_exprs = [
                a.expression for a in self._action_history[:-1]
                if a.action_type == "calculate" and a.expression
            ]
            if action.expression in prev_exprs:
                return -0.05  # Duplicate calculate
            # Try to evaluate expression
            try:
                result = float(eval(action.expression, {"__builtins__": {}}, {}))
            except Exception:
                return -0.02  # Syntax error or invalid expression
            # Check if result is close to expected_result (1% tolerance)
            if action.expected_result is not None:
                tolerance = abs(action.expected_result) * 0.01 if action.expected_result != 0 else 0.01
                if abs(result - action.expected_result) <= tolerance:
                    return +0.05  # Correct arithmetic
            return 0.001  # Evaluated but result doesn't match expected

        elif action.action_type == "flag_compliance_risk":
            has_trap    = gt.get("has_compliance_trap", False)
            trap_section = gt.get("compliance_trap_law_section", "")
            if has_trap and action.law_section == trap_section:
                return +0.10  # Correct flag — exact section match
            elif has_trap and action.law_section != trap_section:
                return  0.001  # Tried but wrong section — no credit, no penalty
            else:
                return -0.10  # False alarm — no trap in this scenario

        elif action.action_type == "escalate_to_professional":
            needs_esc = gt.get("needs_escalation", False)
            esc_type  = gt.get("escalation_type", "")
            if not needs_esc:
                return -0.08  # Unnecessary escalation
            if (action.professional_type and
                    action.professional_type.value.upper() == esc_type.upper()):
                return +0.08  # Correct professional type
            return +0.03  # Needed but wrong professional type

        elif action.action_type == "ask_clarification":
            if action.question_text and len(action.question_text) > 10:
                return +0.02  # Gathering information — marginally positive

        return 0.001

    # ── FINAL REWARD DISPATCHER ──────────────────────────────────────────────

    def _compute_final_reward(self) -> float:
        """
        Dispatch to the appropriate grader when the episode ends.
        Called exactly once per episode — when choose_option terminates.

        For multi-step scenarios, this is called after the LAST choose_option.
        The full action_history (all steps) is passed to the grader.
        """
        gt      = self._current_scenario.get("ground_truth", {})
        task_id = self._current_task.task_id

        if task_id == "financial_optimize":
            return grade_easy(
                action_history = self._action_history,
                ground_truth   = gt,
            )
        elif task_id == "tax_planning":
            return grade_medium(
                action_history = self._action_history,
                ground_truth   = gt,
                step_log       = self._step_log,
            )
        elif task_id == "startup_compliance":
            return grade_hard(
                action_history = self._action_history,
                ground_truth   = gt,
            )
        raise RuntimeError(f"No grader registered for task_id '{task_id}'")

    # ── BUILD OBSERVATION ────────────────────────────────────────────────────

    def _build_observation(self, step_number: int = 1) -> ScenarioObservation:
        """
        Build a ScenarioObservation from the current scenario.
        PRIVACY: ground_truth is NEVER included. The ScenarioObservation
        Pydantic model has no ground_truth field — this is enforced architecturally.
        """
        s = self._current_scenario
        gt = s.get("ground_truth", {})  # Loaded but NEVER passed to observation

        # Get options for the current step
        step_sequence = s.get("step_sequence", [])
        if step_sequence:
            # Multi-step: use first step's options initially
            current_step_data = step_sequence[0]
            raw_opts = current_step_data.get("options", [])
            step_prompt = current_step_data.get("prompt", "")
        else:
            # Easy: use scenario-level options
            raw_opts = s.get("options", [])
            step_prompt = None

        options = [FinancialOption(**opt) for opt in raw_opts]

        # Build financial snapshot from initial_financial_snapshot or financial_snapshot
        snap_data = s.get("initial_financial_snapshot") or s.get("financial_snapshot", {})
        snapshot  = FinancialSnapshot(**snap_data)

        return ScenarioObservation(
            scenario_id          = s["id"],
            title                = s["title"],
            category             = FinancialCategory(s.get("category", "personal_finance")),
            context              = s["context"],
            financial_snapshot   = snapshot,
            options              = options,
            step_count           = self._step_count,
            task_id              = self._current_task.task_id,
            done                 = False,
            questions_asked      = [],
            calculations_done    = [],
            compliance_flags     = [],
            running_balance      = snap_data.get("available_cash"),
            current_step_context = step_prompt,
            current_step_number  = step_number,
        )

    # ── STATE ────────────────────────────────────────────────────────────────

    def state(self) -> Dict[str, Any]:
        """
        Return current environment state for debugging and the /state endpoint.
        SECURITY: action_history and ground_truth are deliberately excluded.
        ground_truth must never be accessible via any public endpoint.
        """
        return {
            "current_observation": self._state,
            "step_count":          self._step_count,
            "task_id":             self._current_task.task_id if self._current_task else None,
            "is_done":             self._done,
            "action_count":        len(self._action_history),
            "available_tasks":     [t.model_dump() for t in list_tasks()],
        }