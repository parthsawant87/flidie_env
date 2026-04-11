# server/app.py
# ─────────────────────────────────────────────────────────────────────────────
# FLIDIE FastAPI Server
# OpenEnv compliant: /reset, /step, /state, /health, /tasks, /web
# ─────────────────────────────────────────────────────────────────────────────

import uuid
from typing import Dict

from fastapi                 import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses       import JSONResponse, HTMLResponse
from flidie_environment import FlidieEnv
from models             import FinancialAction, ResetRequest
from tasks              import list_tasks

# ── APP INIT ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "FLIDIE — Finance & Legal Decision Intelligence Environment",
    description = "OpenEnv compliant. Indian tax, GST, SEBI, FEMA scenarios.",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# ── SESSION REGISTRY ──────────────────────────────────────────────────────────

_sessions: Dict[str, FlidieEnv] = {}


def _require_session(session_id: str | None) -> tuple[str, FlidieEnv]:
    if not session_id or session_id not in _sessions:
        raise HTTPException(
            status_code=400,
            detail="Invalid or missing X-Session-ID. Call /reset first.",
        )
    return session_id, _sessions[session_id]


def _get_or_create_session(session_id: str | None) -> tuple[str, FlidieEnv]:
    key = session_id if (session_id and session_id in _sessions) else str(uuid.uuid4())
    if key not in _sessions:
        _sessions[key] = FlidieEnv()
    return key, _sessions[key]


# ── HEALTH ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0", "sessions": len(_sessions)}

# ── TASKS ─────────────────────────────────────────────────────────────────────

@app.get("/tasks")
async def tasks():
    return {"tasks": [t.model_dump() for t in list_tasks()]}


# ── RESET ─────────────────────────────────────────────────────────────────────

@app.post("/reset")
async def reset(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    task_id    = body.get("task_id", "financial_optimize")
    session_id = request.headers.get("X-Session-ID")
    sid, env   = _get_or_create_session(session_id)

    try:
        obs = env.reset(task_id=task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    response = JSONResponse(content=obs.model_dump())
    response.headers["X-Session-ID"] = sid
    return response


# ── STEP ──────────────────────────────────────────────────────────────────────

@app.post("/step")
async def step(request: Request):
    session_id = request.headers.get("X-Session-ID")
    sid, env   = _require_session(session_id)

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


# ── STATE ─────────────────────────────────────────────────────────────────────

@app.get("/state")
async def state(request: Request):
    session_id = request.headers.get("X-Session-ID")
    sid, env   = _require_session(session_id)
    return env.state()


# ── EXCEPTION HANDLER ─────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}: {exc}"},
    )

#------------------------------------------------------------------------
@app.get("/metadata")
async def metadata():
    return {
        "name": "flidie-env",
        "description": "Finance and Legal Decision Intelligence Environment for Indian financial scenarios."
    }

@app.get("/schema")
async def schema():
    return {
        "action":      FinancialAction.model_json_schema(),
        "observation": ScenarioObservation.model_json_schema(),
        "state":       {"type": "object"}
    }

@app.post("/mcp")
async def mcp(request: Request):
    return {"jsonrpc": "2.0", "id": None, "result": {"capabilities": {}}}

#-------------------------------------------------------------------------

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
# ── WEB UI ────────────────────────────────────────────────────────────────────

@app.get("/web", response_class=HTMLResponse)
async def web_ui(request: Request):
    base_url = str(request.base_url).rstrip("/")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FLIDIE — OpenEnv Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}

:root {{
  --bg: #0f1117;
  --panel: #161b27;
  --panel2: #1c2333;
  --border: #2a3348;
  --gold: #f5c518;
  --teal: #00d4aa;
  --red: #ff4d4d;
  --text: #dde3f0;
  --muted: #6b7a99;
  --mono: 'IBM Plex Mono', monospace;
  --sans: 'IBM Plex Sans', sans-serif;
}}

body {{
  background: var(--bg);
  color: var(--text);
  font-family: var(--sans);
  font-size: 13px;
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}}

/* ── TOP BAR ── */
.topbar {{
  background: var(--panel);
  border-bottom: 1px solid var(--border);
  padding: 0 24px;
  height: 48px;
  display: flex;
  align-items: center;
  gap: 16px;
  flex-shrink: 0;
}}
.logo {{
  font-family: var(--mono);
  font-weight: 600;
  font-size: 15px;
  color: var(--gold);
  letter-spacing: -0.02em;
}}
.logo span {{ color: var(--teal); }}
.badge {{
  font-family: var(--mono);
  font-size: 10px;
  background: rgba(0,212,170,.12);
  color: var(--teal);
  border: 1px solid rgba(0,212,170,.3);
  padding: 2px 8px;
  border-radius: 3px;
  letter-spacing: .08em;
}}
.health-dot {{
  width: 7px; height: 7px;
  border-radius: 50%;
  background: #3dd68c;
  box-shadow: 0 0 6px #3dd68c;
  margin-left: auto;
}}
.health-label {{
  font-family: var(--mono);
  font-size: 11px;
  color: #3dd68c;
}}

/* ── MAIN LAYOUT ── */
.layout {{
  display: grid;
  grid-template-columns: 340px 1fr;
  flex: 1;
  overflow: hidden;
}}

/* ── LEFT PANEL ── */
.left {{
  border-right: 1px solid var(--border);
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 28px;
}}

.section-title {{
  font-family: var(--mono);
  font-size: 10px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .14em;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 8px;
}}
.section-title::after {{
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}}

/* Quick Start */
.qs-label {{
  font-size: 11px;
  color: var(--muted);
  margin-bottom: 6px;
  font-weight: 400;
}}
.code-block {{
  background: #0a0d14;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px 14px;
  font-family: var(--mono);
  font-size: 11px;
  line-height: 1.8;
  position: relative;
  margin-bottom: 12px;
}}
.code-block .cm {{ color: #546e7a; font-style: italic; }}
.code-block .kw {{ color: #c792ea; }}
.code-block .st {{ color: #c3e88d; }}
.code-block .fn {{ color: #82aaff; }}
.code-block .nu {{ color: #f78c6c; }}
.copy-btn {{
  position: absolute;
  top: 8px; right: 8px;
  background: var(--panel2);
  border: 1px solid var(--border);
  color: var(--muted);
  font-family: var(--mono);
  font-size: 10px;
  padding: 2px 7px;
  border-radius: 3px;
  cursor: pointer;
  transition: all .15s;
}}
.copy-btn:hover {{ color: var(--teal); border-color: var(--teal); }}

/* Server connect */
.server-row {{
  display: flex;
  gap: 6px;
  align-items: center;
}}
.server-input {{
  flex: 1;
  background: #0a0d14;
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 7px 10px;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text);
  outline: none;
  transition: border .15s;
}}
.server-input:focus {{ border-color: var(--gold); }}
.btn-sm {{
  background: var(--panel2);
  border: 1px solid var(--border);
  color: var(--text);
  font-family: var(--mono);
  font-size: 11px;
  padding: 7px 12px;
  border-radius: 5px;
  cursor: pointer;
  transition: all .15s;
  white-space: nowrap;
}}
.btn-sm:hover {{ border-color: var(--gold); color: var(--gold); }}

/* Contribute */
.contrib-code {{
  background: #0a0d14;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px 14px;
  font-family: var(--mono);
  font-size: 10.5px;
  line-height: 1.9;
  color: #7ec8e3;
}}
.contrib-code .prompt {{ color: var(--gold); user-select: none; }}

/* Task selector */
.task-select {{
  background: #0a0d14;
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 7px 10px;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text);
  width: 100%;
  outline: none;
  cursor: pointer;
}}
.task-select:focus {{ border-color: var(--gold); }}

/* ── RIGHT PANEL ── */
.right {{
  display: flex;
  flex-direction: column;
  overflow: hidden;
}}

/* Episode stats bar */
.stats-bar {{
  background: var(--panel);
  border-bottom: 1px solid var(--border);
  padding: 10px 24px;
  display: flex;
  align-items: center;
  gap: 28px;
  flex-shrink: 0;
}}
.stat {{
  display: flex;
  align-items: center;
  gap: 8px;
}}
.stat-label {{
  font-family: var(--mono);
  font-size: 11px;
  color: var(--muted);
}}
.stat-val {{
  font-family: var(--mono);
  font-size: 13px;
  font-weight: 600;
}}
.stat-val.reward {{ color: var(--gold); }}
.stat-val.done-f {{ color: var(--teal); }}
.stat-val.done-t {{ color: var(--red); }}
.stat-val.steps {{ color: #82aaff; }}
.stat-val.session {{ color: var(--muted); font-size: 10px; font-weight: 400; }}

/* Interaction area */
.interact {{
  flex: 1;
  display: grid;
  grid-template-rows: auto 1fr;
  overflow: hidden;
  padding: 20px 24px;
  gap: 16px;
}}

/* Action builder */
.action-builder {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px 18px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}}
.ab-row {{
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
}}
.ab-label {{
  font-family: var(--mono);
  font-size: 10px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .1em;
  min-width: 90px;
}}
.ab-select, .ab-input {{
  background: #0a0d14;
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 6px 10px;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text);
  outline: none;
  transition: border .15s;
}}
.ab-select {{ cursor: pointer; }}
.ab-select:focus, .ab-input:focus {{ border-color: var(--gold); }}
.ab-input {{ flex: 1; min-width: 180px; }}

/* Textarea */
.msg-area {{
  background: #0a0d14;
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 8px 10px;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text);
  resize: none;
  outline: none;
  width: 100%;
  height: 54px;
  transition: border .15s;
}}
.msg-area:focus {{ border-color: var(--gold); }}

/* Action buttons */
.btn-row {{
  display: flex;
  gap: 8px;
}}
.btn {{
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 500;
  padding: 8px 18px;
  border-radius: 5px;
  border: 1px solid transparent;
  cursor: pointer;
  transition: all .15s;
  letter-spacing: .02em;
}}
.btn-step {{
  background: var(--teal);
  color: #000;
  border-color: var(--teal);
}}
.btn-step:hover {{ background: #00b894; }}
.btn-reset {{
  background: transparent;
  color: var(--text);
  border-color: var(--border);
}}
.btn-reset:hover {{ border-color: var(--gold); color: var(--gold); }}
.btn-state {{
  background: transparent;
  color: var(--text);
  border-color: var(--border);
}}
.btn-state:hover {{ border-color: #82aaff; color: #82aaff; }}

/* Status */
.status-pill {{
  font-family: var(--mono);
  font-size: 11px;
  padding: 4px 10px;
  border-radius: 3px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}}
.status-pill.ok {{ background: rgba(0,212,170,.1); color: var(--teal); border: 1px solid rgba(0,212,170,.25); }}
.status-pill.err {{ background: rgba(255,77,77,.1); color: var(--red); border: 1px solid rgba(255,77,77,.25); }}
.status-pill.idle {{ background: rgba(107,122,153,.1); color: var(--muted); border: 1px solid var(--border); }}

/* JSON Output panel */
.output-panel {{
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
}}
.output-header {{
  padding: 8px 14px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
}}
.output-title {{
  font-family: var(--mono);
  font-size: 10px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .12em;
}}
.output-meta {{
  font-family: var(--mono);
  font-size: 10px;
  color: var(--muted);
}}
.output-body {{
  flex: 1;
  overflow-y: auto;
  padding: 14px 16px;
  font-family: var(--mono);
  font-size: 11.5px;
  line-height: 1.75;
  white-space: pre;
}}
/* JSON syntax colors */
.j-key {{ color: #82aaff; }}
.j-str {{ color: #c3e88d; }}
.j-num {{ color: #f78c6c; }}
.j-bool {{ color: #c792ea; }}
.j-null {{ color: var(--muted); }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}

/* Loading spinner */
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
.spinner {{
  display: inline-block;
  width: 10px; height: 10px;
  border: 2px solid var(--border);
  border-top-color: var(--teal);
  border-radius: 50%;
  animation: spin .6s linear infinite;
}}
</style>
</head>
<body>

<!-- TOP BAR -->
<div class="topbar">
  <div class="logo">FLIDIE<span>.env</span></div>
  <div class="badge">OPENENV</div>
  <div class="badge" style="color:var(--gold);background:rgba(245,197,24,.1);border-color:rgba(245,197,24,.3);">v1.0.0</div>
  <div style="margin-left:auto;display:flex;align-items:center;gap:8px;">
    <div class="health-dot" id="healthDot"></div>
    <span class="health-label" id="healthLabel">checking...</span>
  </div>
</div>

<!-- MAIN LAYOUT -->
<div class="layout">

  <!-- LEFT PANEL -->
  <div class="left">

    <!-- Quick Start -->
    <div>
      <div class="section-title">Quick Start</div>
      <div class="qs-label">Connect from Python using openenv-core:</div>
      <div class="code-block">
        <button class="copy-btn" onclick="copyCode('pycode')">copy</button>
        <div id="pycode"><span class="kw">from</span> openenv <span class="kw">import</span> MyEnvMix, MyEnvAction
<span class="kw">with</span> MyEnvMix.<span class="fn">from_env_url</span>(<span class="st">"{base_url}"</span>) <span class="kw">as</span> env:
    result = env.<span class="fn">step</span>(MyEnvAction.<span class="fn">make</span>(
        action_type=<span class="st">"choose_option"</span>,
        option_id=<span class="st">"A"</span>
    ))</div>
      </div>

      <div class="qs-label" style="margin-top:4px;">Or connect directly to a running server:</div>
      <div class="server-row">
        <input class="server-input" id="serverUrl" value="{base_url}" readonly>
        <button class="btn-sm" onclick="copyUrl()">copy url</button>
      </div>
    </div>

    <!-- Task Selector -->
    <div>
      <div class="section-title">Active Task</div>
      <div class="qs-label">Select task before clicking Reset:</div>
      <select class="task-select" id="taskSelect">
        <option value="financial_optimize">financial_optimize — easy</option>
        <option value="tax_planning">tax_planning — medium</option>
        <option value="startup_compliance">startup_compliance — hard</option>
      </select>
    </div>

    <!-- Contribute -->
    <div>
      <div class="section-title">Contribute</div>
      <div class="qs-label">Submit improvements via pull request on HF Hub:</div>
      <div class="contrib-code">
        <span class="prompt">$ </span>openenv fork groot87/flidie-env<br>
        <span class="prompt">$ </span>git clone &lt;your-fork&gt;<br>
        <span class="prompt">$ </span><span style="color:#546e7a;"># make your changes</span><br>
        <span class="prompt">$ </span>openenv push --repo-id &lt;username&gt;/flidie-env
      </div>

      <div class="qs-label" style="margin-top:12px;">Links:</div>
      <div style="display:flex;flex-direction:column;gap:4px;margin-top:4px;">
        <a href="/docs" target="_blank" style="color:var(--teal);font-family:var(--mono);font-size:11px;text-decoration:none;">→ /docs — Swagger UI</a>
        <a href="/tasks" target="_blank" style="color:var(--teal);font-family:var(--mono);font-size:11px;text-decoration:none;">→ /tasks — Task registry JSON</a>
        <a href="/health" target="_blank" style="color:var(--teal);font-family:var(--mono);font-size:11px;text-decoration:none;">→ /health — Health check</a>
        <a href="https://huggingface.co/spaces/groot87/flidie-env" target="_blank" style="color:var(--gold);font-family:var(--mono);font-size:11px;text-decoration:none;">→ HF Space</a>
      </div>
    </div>

    <!-- Action Reference -->
    <div>
      <div class="section-title">Action Space</div>
      <div style="display:flex;flex-direction:column;gap:5px;">
        <div class="code-block" style="padding:8px 12px;font-size:10.5px;margin-bottom:4px;">
          <span style="color:var(--gold);">choose_option</span> &nbsp;<span style="color:var(--muted);">option_id: A|B|C|D</span>
        </div>
        <div class="code-block" style="padding:8px 12px;font-size:10.5px;margin-bottom:4px;">
          <span style="color:var(--gold);">calculate</span> &nbsp;<span style="color:var(--muted);">expression + expected_result</span>
        </div>
        <div class="code-block" style="padding:8px 12px;font-size:10.5px;margin-bottom:4px;">
          <span style="color:var(--gold);">flag_compliance_risk</span> &nbsp;<span style="color:var(--muted);">law_section + risk_description</span>
        </div>
        <div class="code-block" style="padding:8px 12px;font-size:10.5px;margin-bottom:4px;">
          <span style="color:var(--gold);">escalate_to_professional</span> &nbsp;<span style="color:var(--muted);">CA|lawyer|SEBI_advisor</span>
        </div>
        <div class="code-block" style="padding:8px 12px;font-size:10.5px;margin-bottom:0;">
          <span style="color:var(--gold);">ask_clarification</span> &nbsp;<span style="color:var(--muted);">question_text</span>
        </div>
      </div>
    </div>

  </div><!-- /left -->

  <!-- RIGHT PANEL -->
  <div class="right">

    <!-- Stats bar -->
    <div class="stats-bar">
      <div class="stat">
        <span class="stat-label">Reward:</span>
        <span class="stat-val reward" id="statReward">—</span>
      </div>
      <div class="stat">
        <span class="stat-label">Done:</span>
        <span class="stat-val done-f" id="statDone">—</span>
      </div>
      <div class="stat">
        <span class="stat-label">Step:</span>
        <span class="stat-val steps" id="statStep">—</span>
      </div>
      <div class="stat" style="margin-left:auto;">
        <span class="stat-label">Session:</span>
        <span class="stat-val session" id="statSession">none</span>
      </div>
    </div>

    <!-- Interaction -->
    <div class="interact">

      <!-- Action builder -->
      <div class="action-builder">
        <div class="ab-row">
          <span class="ab-label">Action Type</span>
          <select class="ab-select" id="actionType" onchange="updateFields()">
            <option value="choose_option">choose_option</option>
            <option value="calculate">calculate</option>
            <option value="flag_compliance_risk">flag_compliance_risk</option>
            <option value="escalate_to_professional">escalate_to_professional</option>
            <option value="ask_clarification">ask_clarification</option>
          </select>
        </div>

        <!-- Dynamic fields rendered by JS -->
        <div id="dynamicFields"></div>

        <div class="btn-row">
          <button class="btn btn-step" id="btnStep" onclick="doStep()">▶ Step</button>
          <button class="btn btn-reset" onclick="doReset()">↺ Reset</button>
          <button class="btn btn-state" onclick="getState()">◈ Get State</button>
          <span id="statusPill" class="status-pill idle">idle</span>
        </div>
      </div>

      <!-- Output panel -->
      <div class="output-panel">
        <div class="output-header">
          <span class="output-title">Raw JSON Response</span>
          <span class="output-meta" id="outputMeta">—</span>
        </div>
        <div class="output-body" id="outputBody">
<span style="color:var(--muted);">// Click Reset to start an episode, then Step to take actions.
// Responses appear here as formatted JSON.</span>
        </div>
      </div>

    </div><!-- /interact -->
  </div><!-- /right -->
</div><!-- /layout -->

<script>
let sessionId = null;

// ── HEALTH CHECK ──────────────────────────────────────────────────────────────
async function checkHealth() {{
  try {{
    const r = await fetch('/health');
    const d = await r.json();
    document.getElementById('healthDot').style.background = '#3dd68c';
    document.getElementById('healthDot').style.boxShadow = '0 0 6px #3dd68c';
    document.getElementById('healthLabel').textContent = 'healthy · ' + d.sessions + ' sessions';
  }} catch {{
    document.getElementById('healthDot').style.background = '#ff4d4d';
    document.getElementById('healthDot').style.boxShadow = '0 0 6px #ff4d4d';
    document.getElementById('healthLabel').style.color = '#ff4d4d';
    document.getElementById('healthLabel').textContent = 'unreachable';
  }}
}}
checkHealth();
setInterval(checkHealth, 15000);

// ── DYNAMIC FIELDS ────────────────────────────────────────────────────────────
function updateFields() {{
  const t = document.getElementById('actionType').value;
  const c = document.getElementById('dynamicFields');
  const field = (label, id, placeholder, extra='') =>
    `<div class="ab-row">
      <span class="ab-label">${{label}}</span>
      <input class="ab-input" id="${{id}}" placeholder="${{placeholder}}" ${{extra}}>
    </div>`;
  const sel = (label, id, opts) =>
    `<div class="ab-row">
      <span class="ab-label">${{label}}</span>
      <select class="ab-select" id="${{id}}">${{opts.map(o=>`<option value="${{o}}">${{o}}</option>`).join('')}}</select>
    </div>`;

  const map = {{
    choose_option: sel('Option ID', 'f_option', ['A','B','C','D']),
    calculate: field('Expression', 'f_expr', '90000 * 0.3') +
               field('Expected', 'f_expected', '27000.0'),
    flag_compliance_risk: field('Law Section', 'f_law', 'CGST_ACT_2017_S22') +
                          field('Risk Description', 'f_risk', 'Explain the risk...'),
    escalate_to_professional: sel('Professional', 'f_prof', ['CA','lawyer','SEBI_advisor']),
    ask_clarification: field('Question', 'f_question', 'What is the applicable tax bracket?'),
  }};
  c.innerHTML = map[t] || '';
}}
updateFields();

// ── BUILD ACTION PAYLOAD ───────────────────────────────────────────────────────
function buildAction() {{
  const t = document.getElementById('actionType').value;
  const v = id => {{ const el = document.getElementById(id); return el ? el.value : null; }};
  const base = {{ action_type: t }};
  if (t === 'choose_option')            return {{ ...base, option_id: v('f_option') }};
  if (t === 'calculate')                return {{ ...base, expression: v('f_expr'), expected_result: parseFloat(v('f_expected')) || null }};
  if (t === 'flag_compliance_risk')     return {{ ...base, law_section: v('f_law'), risk_description: v('f_risk') }};
  if (t === 'escalate_to_professional') return {{ ...base, professional_type: v('f_prof') }};
  if (t === 'ask_clarification')        return {{ ...base, question_text: v('f_question') }};
  return base;
}}

// ── API CALLS ─────────────────────────────────────────────────────────────────
function setStatus(msg, type) {{
  const el = document.getElementById('statusPill');
  el.className = 'status-pill ' + type;
  el.textContent = msg;
}}

function renderJSON(obj) {{
  return JSON.stringify(obj, null, 2)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"([^"]+)":/g, '<span class="j-key">"$1"</span>:')
    .replace(/: "([^"]*)"/g, ': <span class="j-str">"$1"</span>')
    .replace(/: (\\d+\\.?\\d*)/g, ': <span class="j-num">$1</span>')
    .replace(/: (true|false)/g, ': <span class="j-bool">$1</span>')
    .replace(/: null/g, ': <span class="j-null">null</span>');
}}

function showOutput(data, meta) {{
  document.getElementById('outputBody').innerHTML = renderJSON(data);
  document.getElementById('outputMeta').textContent = meta;
}}

function updateStats(data) {{
  if ('reward' in data) {{
    document.getElementById('statReward').textContent = 
      typeof data.reward === 'number' ? data.reward.toFixed(2) : data.reward;
  }}
  if ('done' in data) {{
    const el = document.getElementById('statDone');
    el.textContent = String(data.done);
    el.className = 'stat-val ' + (data.done ? 'done-t' : 'done-f');
  }}
  if ('step_count' in data)
    document.getElementById('statStep').textContent = data.step_count;
  if (sessionId)
    document.getElementById('statSession').textContent = sessionId.slice(0,8) + '...';
}}

async function doReset() {{
  const task = document.getElementById('taskSelect').value;
  setStatus('resetting...', 'idle');
  const t0 = Date.now();
  try {{
    const r = await fetch('/reset', {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{task_id: task}})
    }});
    sessionId = r.headers.get('x-session-id');
    const data = await r.json();
    document.getElementById('statReward').textContent = '—';
    document.getElementById('statDone').textContent = 'false';
    document.getElementById('statDone').className = 'stat-val done-f';
    document.getElementById('statStep').textContent = '0';
    document.getElementById('statSession').textContent = sessionId ? sessionId.slice(0,8)+'...' : 'none';
    showOutput(data, `POST /reset · ${{Date.now()-t0}}ms · ${{task}}`);
    setStatus('episode started', 'ok');
  }} catch(e) {{
    showOutput({{error: e.message}}, 'POST /reset · error');
    setStatus('reset failed', 'err');
  }}
}}

async function doStep() {{
  if (!sessionId) {{ setStatus('call Reset first', 'err'); return; }}
  const action = buildAction();
  setStatus('stepping...', 'idle');
  const t0 = Date.now();
  try {{
    const r = await fetch('/step', {{
      method: 'POST',
      headers: {{'Content-Type':'application/json','x-session-id':sessionId}},
      body: JSON.stringify(action)
    }});
    const data = await r.json();
    updateStats(data);
    showOutput(data, `POST /step · ${{Date.now()-t0}}ms · ${{action.action_type}}`);
    setStatus(data.done ? 'episode done ✓' : 'step complete', data.done ? 'err' : 'ok');
  }} catch(e) {{
    showOutput({{error: e.message}}, 'POST /step · error');
    setStatus('step failed', 'err');
  }}
}}

async function getState() {{
  if (!sessionId) {{ setStatus('call Reset first', 'err'); return; }}
  setStatus('fetching state...', 'idle');
  const t0 = Date.now();
  try {{
    const r = await fetch('/state', {{headers:{{'x-session-id':sessionId}}}});
    const data = await r.json();
    updateStats(data);
    showOutput(data, `GET /state · ${{Date.now()-t0}}ms`);
    setStatus('state loaded', 'ok');
  }} catch(e) {{
    showOutput({{error: e.message}}, 'GET /state · error');
    setStatus('state failed', 'err');
  }}
}}

function copyCode(id) {{
  const txt = document.getElementById(id).innerText;
  navigator.clipboard.writeText(txt);
}}

function copyUrl() {{
  navigator.clipboard.writeText(document.getElementById('serverUrl').value);
}}
</script>
</body>
</html>"""