# server/app.py
# ─────────────────────────────────────────────────────────────────────────────
# FLIDIE FastAPI Server
#
# Exposes the three OpenEnv-required endpoints: /reset, /step, /state
# Plus /health and /tasks for operational needs.
#
# Session registry: each HTTP session gets its own FlidieEnv instance.
# This is critical for concurrent evaluation — multiple judges may call
# the environment simultaneously, and shared state would corrupt results.
# ─────────────────────────────────────────────────────────────────────────────

import uuid
from typing import Dict

from fastapi                  import FastAPI, HTTPException, Request
from fastapi.middleware.cors  import CORSMiddleware
from fastapi.responses        import JSONResponse

from flidie_environment import FlidieEnv
from models             import FinancialAction, ResetRequest
from tasks              import list_tasks

# ── APP INIT ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "FLIDIE — Finance & Legal Decision Intelligence Environment",
    description = "OpenEnv compliant. Indian tax, GST, SEBI, FEMA scenarios.",
    version     = "1.0.0",
)

# CORS: required for the OpenEnv web dashboard (/web) to call your environment
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# ── SESSION REGISTRY ─────────────────────────────────────────────────────────
# Maps session_id (UUID string) → FlidieEnv instance.
# Each /reset creates or reuses a session. Sessions persist until server restarts.

_sessions: Dict[str, FlidieEnv] = {}


def _get_or_create_session(session_id: str | None) -> tuple[str, FlidieEnv]:
    key = session_id if session_id else "default"
    if key not in _sessions:
        _sessions[key] = FlidieEnv()
    return key, _sessions[key]


# ── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Health check. Docker HEALTHCHECK and HF Spaces use this.
    Returns 200 with status info if server is alive.
    """
    return {
        "status":   "ok",
        "version":  "1.0.0",
        "sessions": len(_sessions),
    }


@app.get("/tasks")
async def tasks():
    """
    List all available tasks with metadata.
    Called by openenv validate and inference scripts.
    """
    return {"tasks": [t.model_dump() for t in list_tasks()]}


@app.post("/reset")
async def reset(request: Request):
    """
    Start a new episode. Returns the initial observation + session_id header.

    Body (JSON): {"task_id": "financial_optimize"}
    Header (optional): X-Session-ID to reuse an existing session

    The X-Session-ID in the response header must be included
    in all subsequent /step calls to maintain episode isolation.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    task_id    = body.get("task_id", "financial_optimize")
    session_id = request.headers.get("X-Session-ID")

    sid, env = _get_or_create_session(session_id)

    try:
        obs = env.reset(task_id=task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    response = JSONResponse(content=obs.model_dump())
    response.headers["X-Session-ID"] = sid
    return response


@app.post("/step")
async def step(request: Request):
    session_id = request.headers.get("X-Session-ID")
    sid, env = _get_or_create_session(session_id)

    try:
        body   = await request.json()
        action = FinancialAction.model_validate(body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")

    try:
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result.model_dump()


@app.get("/state")
async def state(request: Request):
    """
    Inspect the current environment state without modifying it.
    Called by openenv validate to verify spec compliance.
    SECURITY: ground_truth is never accessible via this endpoint.
    Returns minimal valid state if no session is active.
    """
    session_id = request.headers.get("X-Session-ID")
    if not session_id or session_id not in _sessions:
        return {
            "current_observation": {},
            "step_count":          0,
            "task_id":             None,
            "is_done":             False,
            "action_count":        0,
            "available_tasks":     [t.model_dump() for t in list_tasks()],
        }
    return _sessions[session_id].state()


# ── EXCEPTION HANDLER ────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Catch-all: returns JSON error instead of HTML 500 page.
    FastAPI's default 500 response is HTML, which breaks JSON-parsing clients.
    """
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}: {exc}"},
    )
