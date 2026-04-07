# CLAUDE.md — FLIDIE Project Guide

## Project: FLIDIE OpenEnv Environment
Finance & Legal Decision Intelligence Environment for Indian financial scenarios.
OpenEnv hackathon submission (Meta × PyTorch × Hugging Face).

## Critical File Locations
ALL core Python files are at PROJECT ROOT (not in subdirectories):
- models.py          ← Pydantic v2 contracts
- graders.py         ← grade_easy(), grade_medium(), grade_hard()
- tasks.py           ← TASK_REGISTRY, get_task(), list_tasks()
- flidie_environment.py  ← FlidieEnvironment class
- inference.py       ← Baseline agent script (MANDATORY for submission)

Server files are in server/:
- server/__init__.py
- server/app.py      ← FastAPI endpoints /reset /step /state /health

Data files are in data/:
- data/scenarios_easy.json
- data/scenarios_medium.json
- data/scenarios_hard.json

## Import Style — Always Use Flat Imports
CORRECT:   from models import FinancialAction
INCORRECT: from env.models import FinancialAction

## Task IDs — Locked Values
These three strings appear in openenv.yaml, tasks.py, and all reset() calls.
Never change them without updating all three locations:
- "financial_optimize"   (easy)
- "tax_planning"         (medium)
- "startup_compliance"   (hard)

## Pydantic Version: v2 ONLY
Use .model_dump() not .dict()
Use .model_validate() not .parse_obj()
pydantic==2.7.1 is pinned in requirements.txt

## Privacy Rule — Never Violate
ground_truth, outcome_map, optimal_option, compliance_trap_law_section
must NEVER appear in any ScenarioObservation, StepResult, or HTTP response.
The ScenarioObservation Pydantic model has no ground_truth field —
this is enforced architecturally.

## Port Convention
Local dev (no Docker): uvicorn on 8000
Local dev (Docker): docker run -p 8000:7860 flidie-env
HF Spaces: container runs on 7860 (hardcoded in Dockerfile and CMD)
