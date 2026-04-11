# models.py — AT PROJECT ROOT
# ─────────────────────────────────────────────────────────────────────────────
# FLIDIE Pydantic v2 Models
# Finance & Legal Decision Intelligence Environment
#
# Architecture guarantee: ScenarioObservation has NO ground_truth field.
# Ground truth is structurally impossible to leak — the Pydantic model
# rejects it at validation time before it could reach an HTTP response.
# ─────────────────────────────────────────────────────────────────────────────

from pydantic import BaseModel, Field, ConfigDict
from typing   import Optional, Dict, Any, List, Literal
from enum     import Enum


# ── ENUMS ─────────────────────────────────────────────────────────────────────

class OutcomeTier(str, Enum):
    """
    Six reward tiers. Each maps to a real financial consequence under Indian law.
    The grader assigns one tier per decision. Deterministic. No LLM calls.

    optimal  -> +1.00  Best legal outcome, maximum benefit, correct law applied
    good     -> +0.75  Correct choice, incomplete reasoning or minor suboptimality
    neutral  -> +0.40  Acceptable but leaves money on the table
    bad      -> +0.10  Wrong choice that misunderstands basic tax structure
    harmful  -> -0.40  Actively damages client's position, creates penalty exposure
    illegal  -> -1.00  Violates Indian statute — IT Act / GST Act / SEBI / FEMA
    """
    optimal = "optimal"
    good    = "good"
    neutral = "neutral"
    bad     = "bad"
    harmful = "harmful"
    illegal = "illegal"


class FinancialCategory(str, Enum):
    personal_finance = "personal_finance"
    investments      = "investments"
    tax_gst          = "tax_gst"
    legal            = "legal"
    startup          = "startup"
    risk             = "risk"


class ProfessionalType(str, Enum):
    """Who the agent can escalate to. Maps to compliance domain."""
    ca           = "CA"           # Chartered Accountant — tax, GST, audit
    lawyer       = "lawyer"      # Legal counsel — contracts, FEMA, SEBI
    sebi_advisor = "SEBI_advisor" # SEBI-registered investment advisor


# ── SUPPORTING MODELS ─────────────────────────────────────────────────────────

class FinancialOption(BaseModel):
    """One of the four A/B/C/D choices presented to the agent per step."""
    id:    Literal["A", "B", "C", "D"]
    text:  str  # Full description of the choice
    brief: str  # One-line summary of consequence


class FinancialSnapshot(BaseModel):
    """
    Structured financial context for the scenario.
    All fields Optional — not every scenario uses all fields.
    Scenario-specific fields go in extra_fields dict.
    """
    # Income & Tax
    annual_income:           Optional[float] = None
    monthly_income:          Optional[float] = None
    tax_regime:              Optional[str]   = None   # "old" | "new"
    tax_bracket:             Optional[str]   = None   # "5%" | "20%" | "30%"
    # Deductions (old regime)
    existing_80c:            Optional[float] = None
    deduction_80c:           Optional[float] = None
    deduction_80d:           Optional[float] = None
    deduction_80ccd1b:       Optional[float] = None
    home_loan_24b:           Optional[float] = None
    standard_deduction:      Optional[float] = None
    total_deductions:        Optional[float] = None
    # Cash & Investments
    available_cash:          Optional[float] = None
    monthly_surplus:         Optional[float] = None
    existing_equity:         Optional[float] = None
    emergency_fund_months:   Optional[float] = None
    # Loan
    outstanding_loan:        Optional[float] = None
    loan_interest_rate:      Optional[float] = None
    loan_remaining_years:    Optional[int]   = None
    # Business / GST
    business_type:           Optional[str]   = None
    annual_turnover_fy:      Optional[float] = None
    gst_registration:        Optional[str]   = None
    # Timing
    fiscal_year_end_days:    Optional[int]   = None
    advance_tax_paid:        Optional[float] = None
    current_date:            Optional[str]   = None
    # Catch-all for scenario-specific fields not listed above
    extra_fields: Dict[str, Any] = Field(default_factory=dict)


# ── ACTION — 5 Types ──────────────────────────────────────────────────────────

class FinancialAction(BaseModel):
    """
    Discriminated union — action_type determines which fields are meaningful.

    choose_option          : Terminal. Ends episode, triggers grader.
    ask_clarification      : Get additional context. Costs 1 step.
    calculate              : Verify arithmetic. Grader credits correct calculations.
    flag_compliance_risk   : FLIDIE's unique mechanic. Cite the exact law section
                             of a hidden compliance trap. +0.15 bonus if correct.
                             -0.10 penalty for false alarm (no trap in scenario).
    escalate_to_professional: Recommend CA / lawyer / SEBI advisor.
                             Rewarded in hard cases that need expert advice.
                             Penalized in easy cases (unnecessary overhead).
    """
    action_type: Literal[
        "choose_option",
        "ask_clarification",
        "calculate",
        "flag_compliance_risk",
        "escalate_to_professional",
    ]

    # choose_option fields
    option_id: Optional[Literal["A", "B", "C", "D"]] = None

    # ask_clarification fields
    question_text: Optional[str] = None

    # calculate fields
    # expression: Python arithmetic string, e.g. "1500000 * 0.3"
    # Must match one of the key_calculations strings in ground_truth for credit
    expression:      Optional[str]   = None
    expected_result: Optional[float] = None

    # flag_compliance_risk fields
    # law_section MUST exactly match ground_truth["compliance_trap_law_section"]
    # Use canonical strings: "CGST_ACT_2017_S22", "IT_ACT_1961_S17_2_VI", etc.
    law_section:      Optional[str] = None
    risk_description: Optional[str] = None

    # escalate_to_professional fields
    professional_type: Optional[ProfessionalType] = None

    # Optional — not graded but useful for inference script debugging
    reasoning: Optional[str] = None


# ── OBSERVATION — What the agent sees ─────────────────────────────────────────

class ScenarioObservation(BaseModel):
    model_config = ConfigDict(extra="ignore")   # ← ADD THIS LINE
    """
    Everything the agent sees. Contains ZERO ground_truth fields.

    This is not just convention — the model structurally has no ground_truth
    field, making it impossible for grader internals to appear in any HTTP
    response. The validator scans for "ground_truth", "outcome_map", and
    "optimal_option" in all responses. Zero tolerance.

    Dynamic fields update as episode progresses:
      questions_asked      : grows as agent calls ask_clarification()
      calculations_done    : grows as agent calls calculate()
      compliance_flags     : grows as agent calls flag_compliance_risk()
      running_balance      : changes between steps (medium/hard tasks)
      current_step_context : shows the current step's prompt (medium/hard)
    """
    # Static — set at reset(), never change within an episode
    scenario_id:        str
    title:              str
    category:           FinancialCategory
    context:            str
    financial_snapshot: FinancialSnapshot
    options:            List[FinancialOption]

    # Dynamic — updated on each step()
    step_count:          int  = 0
    task_id:             str  = "financial_optimize"
    done:                bool = False
    questions_asked:     List[str]             = Field(default_factory=list)
    calculations_done:   List[Dict[str, Any]]  = Field(default_factory=list)
    compliance_flags:    List[str]             = Field(default_factory=list)

    # Medium / hard task fields
    running_balance:       Optional[float] = None   # Tracks cash through multi-step
    current_step_context:  Optional[str]   = None   # Current step's prompt
    current_step_number:   int             = 1


# ── STEP RESULT ───────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    """Returned by /step endpoint after every agent action."""
    observation: ScenarioObservation
    reward:      float          = Field(ge=0.01, le=0.99)
    done:        bool
    info:        Dict[str, Any] = Field(default_factory=dict)
    # info includes: step, task_id, action_taken, is_redundant, scenario_id


# ── ENDPOINT MODELS ───────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    """Body for POST /reset"""
    task_id: str = "financial_optimize"


class TaskInfo(BaseModel):
    """One task's metadata. Returned by /tasks and /state."""
    id:          str
    name:        str
    difficulty:  Literal["easy", "medium", "hard"]
    description: str
    max_steps:   int = 8
