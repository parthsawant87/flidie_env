# tasks.py — PROJECT ROOT
# ─────────────────────────────────────────────────────────────────────────────
# FLIDIE Task Registry
# Central configuration. All three graders map from here.
# task_ids here MUST match: openenv.yaml, /reset endpoint, scenario JSON files.
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass
from typing import List, Dict
from server.models      import TaskInfo


@dataclass
class TaskConfig:
    """Complete configuration for one task. Used by the environment class."""
    task_id:    str
    name:       str
    difficulty: str    # "easy" | "medium" | "hard"
    data_file:  str    # path relative to project root, e.g. "data/scenarios_easy.json"
    max_steps:  int    # episode terminates at timeout after this many actions
    description: str


TASK_REGISTRY: Dict[str, TaskConfig] = {

    "financial_optimize": TaskConfig(
        task_id     = "financial_optimize",
        name        = "Financial Decision Optimization",
        difficulty  = "easy",
        data_file   = "data/scenarios_easy.json",
        max_steps   = 6,    # 1 choose_option + up to 2 calculate + 3 buffer
        description = (
            "Choose the optimal financial action from four options. "
            "Scenarios drawn from Indian tax, investment, and personal finance "
            "decisions. Ground truth cites Income Tax Act / SEBI regulations."
        ),
    ),

    "tax_planning": TaskConfig(
        task_id     = "tax_planning",
        name        = "Multi-Step Tax Planning",
        difficulty  = "medium",
        data_file   = "data/scenarios_medium.json",
        max_steps   = 10,   # 3 decision steps + up to 7 calculate/clarify actions
        description = (
            "Navigate a 3-step tax scenario where each decision changes the financial state. "
            "Covers GST ITC calculation, income tax regime comparison, advance tax. "
            "Requires arithmetic verification via calculate() to score maximum points."
        ),
    ),

    "startup_compliance": TaskConfig(
        task_id     = "startup_compliance",
        name        = "Startup Compliance Audit",
        difficulty  = "hard",
        data_file   = "data/scenarios_hard.json",
        max_steps   = 10,   # needs time for trap detection + escalation + calculation
        description = (
            "Audit a startup founder's financial situation where the client believes "
            "they are compliant but a hidden violation may exist. Must identify the "
            "compliance trap, advise correctly, and recommend professional escalation "
            "when needed. Covers ESOP tax, GST registration, FEMA, angel tax."
        ),
    ),
}


def get_task(task_id: str) -> TaskConfig:
    """
    Retrieve TaskConfig by task_id.

    Raises ValueError for unknown task_id with helpful error message listing
    all valid options. Never returns None — caller can always trust the result.
    """
    if task_id not in TASK_REGISTRY:
        valid = list(TASK_REGISTRY.keys())
        raise ValueError(
            f"Unknown task_id '{task_id}'. Valid options: {valid}"
        )
    return TASK_REGISTRY[task_id]


def list_tasks() -> List[TaskInfo]:
    """Return TaskInfo models for all registered tasks. Used by /state and /tasks endpoints."""
    return [
        TaskInfo(
            id          = cfg.task_id,
            name        = cfg.name,
            difficulty  = cfg.difficulty,
            description = cfg.description,
            max_steps   = cfg.max_steps,
        )
        for cfg in TASK_REGISTRY.values()
    ]


def get_all_task_ids() -> List[str]:
    """Return all registered task IDs. Used by inference.py to iterate tasks."""
    return list(TASK_REGISTRY.keys())
