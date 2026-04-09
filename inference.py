# FLIDIE Baseline Inference Script
#
# Usage:
#   python3 inference.py              # uses ENV_URL from .env
#   ENV_URL=http://localhost:8000 python3 inference.py  # override URL
#
# Credentials needed in .env:
#   HF_TOKEN=hf_...                   # Hugging Face write token
#   API_BASE_URL=https://router.huggingface.co/v1
#   MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
#
# What this does:
#   1. Calls /reset to start an episode for each task
#   2. Sends observation to Qwen-72B via HF Router (free, no paid API)
#   3. Parses LLM JSON response into FinancialAction
#   4. Calls /step with that action
#   5. Repeats until done=True
#   6. Prints final score per task
#   7. Prints summary table — COPY THESE SCORES INTO README.md
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import textwrap
import requests
from openai   import OpenAI
from dotenv   import load_dotenv
from typing   import Optional


load_dotenv()


# ── CONFIGURATION ─────────────────────────────────────────────────────────────


HF_TOKEN = os.environ.get("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",  "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("ENV_URL",     "https://huggingface.co/spaces/groot87/flidie-env").rstrip("/")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN is not set.\n"
        "Add it to your .env file: HF_TOKEN=hf_your_token_here\n"
        "Get a token at: huggingface.co → Settings → Access Tokens"
    )

# OpenAI-compatible client pointed at HF Router
# The HF Router speaks the OpenAI API protocol.
# Using HF_TOKEN as the API key — no OpenAI account needed.
client = OpenAI(
    base_url   = API_BASE_URL,
    api_key    = HF_TOKEN,
)

TASK_IDS = [
    "financial_optimize",
    "tax_planning",
    "startup_compliance",
]

EPISODES_PER_TASK = 3   # Run each task 3 times, average the scores
MAX_STEPS         = 20  # Safety cap — environment has its own limit but this prevents infinite loops


# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
# This is what tells the LLM how to behave in FLIDIE.
# It explains the action space so the LLM knows exactly what JSON to output.
# Keep it concise — the observation (scenario context) is in the user message.

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert Indian financial and legal advisor.
You are given a financial scenario and must take actions to advise a client correctly.

You must respond with ONLY a valid JSON object — no explanation, no markdown, no preamble.

Available actions (pick exactly one per turn):

1. Ask a clarifying question:
   {"action_type": "ask_clarification", "question_text": "your question here"}

2. Perform a financial calculation before deciding:
   {"action_type": "calculate", "expression": "90000 * 0.3", "expected_result": 27000.0}

3. Flag a compliance risk with the exact law section:
   {"action_type": "flag_compliance_risk", "law_section": "CGST_ACT_2017_S22", "risk_description": "why"}

4. Escalate to a professional:
   {"action_type": "escalate_to_professional", "professional_type": "CA"}
   (professional_type must be one of: CA, lawyer, SEBI_advisor)

5. Make your final decision (use this when ready to commit):
   {"action_type": "choose_option", "option_id": "A"}
   (option_id must be one of: A, B, C, D)

Strategy for hard scenarios:
- Read the scenario carefully for hidden compliance issues
- Use calculate() to verify arithmetic before deciding
- Use flag_compliance_risk() if you spot a law violation
- Use escalate_to_professional() when the situation requires expert help
- End with choose_option() when confident

You are scored on the quality of your final decision, your calculations, and your compliance awareness.
""").strip()


# ── ENVIRONMENT API ───────────────────────────────────────────────────────────

def env_reset(task_id: str) -> dict:
    """POST /reset → initial observation dict."""
    r = requests.post(
        f"{ENV_URL}/reset",
        json    = {"task_id": task_id},
        timeout = 30,
    )
    r.raise_for_status()
    return r.json()


def env_step(action: dict) -> dict:
    """POST /step → StepResult dict with observation, reward, done."""
    r = requests.post(
        f"{ENV_URL}/step",
        json    = action,
        timeout = 30,
    )
    r.raise_for_status()
    return r.json()


# ── LLM REASONING ─────────────────────────────────────────────────────────────

def build_user_message(obs: dict, step: int, task_id: str) -> str:
    """
    Convert the environment observation into a human-readable prompt.
    The LLM sees this as the "user turn" in the conversation.
    """
    snap = obs.get("financial_snapshot", {})

    # Build financial data summary
    snap_lines = []
    for k, v in snap.items():
        if k == "extra_fields" and isinstance(v, dict):
            for ek, ev in v.items():
                snap_lines.append(f"  {ek}: {ev}")
        elif v is not None:
            snap_lines.append(f"  {k}: {v}")

    # Build options list
    options_text = "\n".join(
        f"  {o['id']}: {o['text']}\n     → {o['brief']}"
        for o in obs.get("options", [])
    )

    # Build trajectory history (what agent has done so far)
    history_parts = []
    if obs.get("calculations_done"):
        history_parts.append("Calculations done this episode:")
        for c in obs["calculations_done"]:
            match = "✓" if c.get("matches_expected") else "?"
            history_parts.append(f"  {c['expression']} = {c['result']} {match}")
    if obs.get("compliance_flags"):
        history_parts.append(f"Compliance flags raised: {obs['compliance_flags']}")
    if obs.get("questions_asked"):
        history_parts.append(f"Questions asked: {obs['questions_asked']}")

    history_text = ("\n".join(history_parts) + "\n") if history_parts else ""

    return textwrap.dedent(f"""
SCENARIO: {obs.get('title', 'Financial Decision')}
Category: {obs.get('category', '')} | Task: {task_id} | Step: {step}

SITUATION:
{obs.get('context', '')}

FINANCIAL DATA:
{chr(10).join(snap_lines) if snap_lines else '  (no structured data)'}

YOUR OPTIONS:
{options_text}

{history_text}What is your next action? Respond with a single JSON object.
""").strip()


def get_llm_action(obs: dict, step: int, task_id: str) -> Optional[dict]:
    """
    Call the LLM and parse its response into a FinancialAction dict.
    Returns None if the LLM output cannot be parsed.

    On parse failure, falls back to a safe default action based on the step:
    - If early in episode: ask_clarification
    - If near max steps: choose_option A (better than timeout at -0.30)
    """
    user_msg = build_user_message(obs, step, task_id)

    try:
        response = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature = 0.1,   # Low temperature for more consistent JSON output
            max_tokens  = 300,   # Action JSON is small — 300 tokens is generous
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if model adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()

        action = json.loads(raw)
        return action

    except json.JSONDecodeError:
        print(f"    [parse fail] LLM output was not valid JSON. Using fallback.")
        return {"action_type": "choose_option", "option_id": "A"}

    except Exception as e:
        print(f"    [LLM error] {e}")
        return {"action_type": "choose_option", "option_id": "A"}


# ── EPISODE RUNNER ────────────────────────────────────────────────────────────

def run_episode(task_id: str, episode_num: int) -> float:
    """
    Run one complete episode for a given task.
    Returns the final reward (float in [-1.0, 1.0]).

    Episode flow:
    1. Reset environment → get initial observation
    2. Get LLM action for current observation
    3. Step environment with that action
    4. If done=True: return reward
    5. Else: repeat from step 2 with new observation
    6. Safety: stop after MAX_STEPS even if not done (returns -0.30 from env)
    """
    # ── Required output: one [START] line at episode begin ──────────────────
    print(f"[START] task={task_id} env=flidie model={MODEL_NAME}")

    step_rewards = []   # collect per-step rewards for [END] line
    final_reward  = -0.30
    step          = 0
    result        = {}

    try:
        obs = env_reset(task_id)

        for step in range(1, MAX_STEPS + 1):
            action      = get_llm_action(obs, step, task_id)
            if action is None:
                action  = {"action_type": "choose_option", "option_id": "A"}

            action_type = action.get("action_type", "?")

            try:
                result  = env_step(action)
            except requests.HTTPError as e:
                # Pydantic validation error — bad action shape. Fall back to choose_option.
                fallback = {"action_type": "choose_option", "option_id": "A"}
                result   = env_step(fallback)
                action_type = "choose_option"

            reward  = result.get("reward", 0.0)
            done    = result.get("done", False)
            error   = result.get("last_action_error") or "null"
            obs     = result.get("observation", obs)

            step_rewards.append(reward)
            final_reward = reward

            # ── Required output: one [STEP] line per step ──────────────────
            print(
                f"[STEP] step={step} action={action_type} "
                f"reward={reward:.2f} done={str(done).lower()} error={error}"
            )

            if done:
                break

    finally:
        # ── Required output: one [END] line — always emitted even on exception
        rewards_str = ",".join(f"{r:.2f}" for r in step_rewards)
        success     = str(final_reward > 0).lower()
        actual_steps = len(step_rewards)
        print(f"[END] success={success} steps={actual_steps} rewards={rewards_str}")

    return final_reward


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FLIDIE Baseline Inference")
    print(f"  Environment: {ENV_URL}")
    print(f"  Model:       {MODEL_NAME}")
    print(f"  Episodes:    {EPISODES_PER_TASK} per task")
    print("=" * 60)

    # Verify environment is reachable before running any episodes
    try:
        health = requests.get(f"{ENV_URL}/health", timeout=10)
        health.raise_for_status()
        print(f"Health check: OK ({health.json()})")
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach environment at {ENV_URL}\n"
            f"Error: {e}\n"
            f"Is the server running? Start it with: uvicorn server.app:app --port 8000"
        )

    results = {}

    for task_id in TASK_IDS:
        print(f"\n{'─' * 60}")
        print(f"TASK: {task_id.upper()}")
        print("─" * 60)

        episode_scores = []
        for ep in range(1, EPISODES_PER_TASK + 1):
            score = run_episode(task_id, ep)
            episode_scores.append(score)

        avg = sum(episode_scores) / len(episode_scores)
        results[task_id] = {
            "episodes": episode_scores,
            "average": round(avg, 4),
        }
        print(f"\n  {task_id}: episodes={episode_scores}, avg={avg:.4f}")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BASELINE SCORES — COPY THESE INTO README.md")
    print("=" * 60)
    for task_id, data in results.items():
        avg   = data["average"]
        label = "easy" if task_id == "financial_optimize" \
                else ("medium" if task_id == "tax_planning" else "hard")
        print(f"  {task_id:<30} ({label:<6}) avg={avg:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()