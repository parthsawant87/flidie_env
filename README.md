---
title: FLIDIE Finance Legal Decision Intelligence Environment
emoji: 💰
colorFrom: indigo
colorTo: yellow
sdk: docker
pinned: false
tags:
  - openenv
---

## Why FLIDIE

300 million+ Indian taxpayers make tax optimization, GST compliance, and
investment decisions every year. Indian personal finance AI is one of the
most underdeveloped applications in the RL ecosystem. FLIDIE fills this gap.

**The key differentiator:** Every reward score corresponds to a real legal
consequence under Indian statute. A score of -1.0 means the agent recommended
something for which the client could face criminal prosecution under Section
276C of the Income Tax Act. The environment doesn't just reward correctness
— it enforces the law.

---

## Three Tasks

| Task ID | Difficulty | Domain | Max Steps |
|---|---|---|---|
| `financial_optimize` | Easy | Tax / Investment | 6 |
| `tax_planning` | Medium | Multi-step Tax | 10 |
| `startup_compliance` | Hard | Legal + GST + SEBI | 10 |

### financial_optimize (Easy)
Single-step financial decision. Agent sees a complete scenario (income,
deductions, time pressure) and chooses from 4 options. Scored on which
of 6 reward tiers the choice falls into. Optional calculate() actions
earn a bonus of up to +0.10.

### tax_planning (Medium)
Three-step tax planning scenario. Each step's outcome changes the running
financial balance shown in subsequent observations. Agent should verify
arithmetic with calculate() before committing. Weighted 40/35/25 across steps.

### startup_compliance (Hard)
Startup founder believes they are compliant. A hidden legal violation may
exist under CGST Act, IT Act, SEBI regulations, or FEMA. Agent is scored
across four independent dimensions: decision correctness (40%), compliance
trap detection (25%), professional escalation (20%), calculation quality (15%).

---

## Action Space

| Action | Fields | Effect |
|---|---|---|
| `choose_option` | `option_id: A\|B\|C\|D` | Terminal — ends episode, triggers grader |
| `calculate` | `expression: str`, `expected_result: float` | Verifies arithmetic, updates observation |
| `flag_compliance_risk` | `law_section: str`, `risk_description: str` | Identifies legal trap — bonus if correct |
| `escalate_to_professional` | `professional_type: CA\|lawyer\|SEBI_advisor` | Recommends expert referral |
| `ask_clarification` | `question_text: str` | Gathers information — small positive reward |

---

## Reward Design — 6 Tiers

| Tier | Score | Real-World Meaning |
|---|---|---|
| optimal | +0.9950 | Best legal outcome. Maximum client benefit. |
| good | +0.75 | Correct choice, minor suboptimality. |
| neutral | +0.40 | Acceptable but leaves money on table. |
| bad | +0.10 | Wrong choice, misunderstands tax structure. |
| harmful | -0.40 | Creates penalty exposure for client. |
| **illegal** | **-0.9950** | **Violates Indian statute. Criminal liability.** |

---

## Observation Space

Each observation contains:
- `scenario_id`, `title`, `category`: Scenario metadata
- `context`: The financial situation in plain language
- `financial_snapshot`: Structured financial data (income, deductions, etc.)
- `options`: List of 4 choices (A/B/C/D) with descriptions
- `calculations_done`: Accumulates as agent calls calculate()
- `compliance_flags`: Accumulates as agent calls flag_compliance_risk()
- `questions_asked`: Accumulates as agent calls ask_clarification()
- `running_balance`: Changes between steps in multi-step scenarios

Ground truth is **never included** in any observation. The Pydantic model
enforces this structurally — ScenarioObservation has no ground_truth field.

---

## Baseline Scores (Qwen/Qwen2.5-72B-Instruct, 3 episodes each)

| Task | Difficulty | Avg Score |
|---|---|---|
| financial_optimize | Easy | +0.9950 |
| tax_planning | Medium | +0.5667 |
| startup_compliance | Hard | +0.4000 |


---

## Setup & Usage

```bash
# Clone and install
git clone https://huggingface.co/spaces/YOUR_USERNAME/flidie-env
cd flidie-env
uv sync --extra dev

# Set credentials
cp .env.example .env
# Edit .env — add your HF_TOKEN

# Run tests
python3 -m pytest test_graders.py -v

# Start server
uvicorn server.app:app --port 8000

# Run baseline
python3 inference.py
```

## Docker

```bash
docker build -t flidie-env .
docker run --rm -p 8000:7860 flidie-env
curl http://localhost:8000/health
```

---

## Legal Sources

All scenarios are grounded in Indian law:
- Income Tax Act 1961 (Sections 80C, 80D, 276C, 17(2)(vi))
- CGST Act 2017 (Sections 16, 17(5), 22, 122, 132)
- SEBI Regulations (Insider Trading, Investment Advisers)
- FEMA 2000 (Foreign Exchange Management)

Ground truth cites the specific law section for every scenario.
