#!/bin/bash
# validate-submission.sh
# ─────────────────────────────────────────────────────────────────────────────
# Run from project root: bash validate-submission.sh
# All checks must pass (exit 0) before submitting.
# ─────────────────────────────────────────────────────────────────────────────

set -e
PASS=0
FAIL=0
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # no color

check() {
  if eval "$2" &>/dev/null; then
    echo -e "${GREEN}✓ $1${NC}"
    PASS=$((PASS+1))
  else
    echo -e "${RED}✗ $1${NC}"
    FAIL=$((FAIL+1))
  fi
}

warn() {
  echo -e "${YELLOW}! $1${NC}"
}

echo "────────────────────────────────────────────"
echo " FLIDIE Submission Validator"
echo "────────────────────────────────────────────"
echo ""

echo "── File Structure ──────────────────────────"
check "models.py at root"              "test -f models.py"
check "graders.py at root"             "test -f graders.py"
check "tasks.py at root"               "test -f tasks.py"
check "flidie_environment.py at root"  "test -f flidie_environment.py"
check "inference.py at root"           "test -f inference.py"
check "Dockerfile at root"             "test -f Dockerfile"
check "openenv.yaml at root"           "test -f openenv.yaml"
check "server/app.py exists"           "test -f server/app.py"
check "data/scenarios_easy.json"       "test -f data/scenarios_easy.json"
check "data/scenarios_medium.json"     "test -f data/scenarios_medium.json"
check "data/scenarios_hard.json"       "test -f data/scenarios_hard.json"
check ".env.example exists"            "test -f .env.example"
echo ""

echo "── Security Checks ─────────────────────────"
check ".env is gitignored"             "grep -q '^\.env$' .gitignore"
check ".env is dockerignored"          "grep -q '^\.env$' .dockerignore"
check "HF_TOKEN not in any .py file"   "! grep -r 'hf_' --include='*.py' . 2>/dev/null | grep -v '.env'"
echo ""

echo "── Python Import Tests ─────────────────────"
check "models.py imports"              ".venv/Scripts/python -c 'from models import FinancialAction, ScenarioObservation'"
check "graders.py imports"             ".venv/Scripts/python -c 'from graders import grade_easy, grade_medium, grade_hard, TIER_REWARD'"
check "tasks.py imports (Dict fix)"    ".venv/Scripts/python -c 'from tasks import get_task, list_tasks, TASK_REGISTRY'"
check "flidie_environment.py imports"  ".venv/Scripts/python -c 'from flidie_environment import FlidieEnvironment'"
check "server.app imports"             ".venv/Scripts/python -c 'from server.app import app'"
echo ""

echo "── Task ID Consistency ─────────────────────"
check "task: financial_optimize"       ".venv/Scripts/python -c 'from tasks import get_task; get_task(\"financial_optimize\")'"
check "task: tax_planning"             ".venv/Scripts/python -c 'from tasks import get_task; get_task(\"tax_planning\")'"
check "task: startup_compliance"       ".venv/Scripts/python -c 'from tasks import get_task; get_task(\"startup_compliance\")'"
check "openenv.yaml has financial_optimize" "grep -q 'financial_optimize' openenv.yaml"
check "openenv.yaml has tax_planning"  "grep -q 'tax_planning' openenv.yaml"
check "openenv.yaml has startup_compliance" "grep -q 'startup_compliance' openenv.yaml"
echo ""

echo "── Grader Tests ────────────────────────────"
check "20 grader tests pass"           ".venv/Scripts/python -m pytest test_graders.py -q"
echo ""

echo "── JSON Data Files ─────────────────────────"
check "scenarios_easy.json valid JSON + 10 scenarios" \
  ".venv/Scripts/python -c \"import json; d=json.load(open('data/scenarios_easy.json')); assert len(d)>=10\""
check "scenarios_medium.json valid JSON + 10 scenarios" \
  ".venv/Scripts/python -c \"import json; d=json.load(open('data/scenarios_medium.json')); assert len(d)>=10\""
check "scenarios_hard.json valid JSON + 10 scenarios" \
  ".venv/Scripts/python -c \"import json; d=json.load(open('data/scenarios_hard.json')); assert len(d)>=10\""
check "Hard escalation mix (not all true)" \
  ".venv/Scripts/python -c \"import json; d=json.load(open('data/scenarios_hard.json')); t=sum(c['ground_truth']['needs_escalation'] for c in d); f=len(d)-t; assert t>=5 and f>=2\""
echo ""

echo "── Episode Integration Test ────────────────"
check "Full episode runs end-to-end" \
  ".venv/Scripts/python -c \"
from flidie_environment import FlidieEnvironment
from models import FinancialAction
env = FlidieEnvironment()
obs = env.reset('financial_optimize')
r = env.step(FinancialAction(action_type='choose_option', option_id='A'))
assert r.done
assert -1.0 <= r.reward <= 1.0
\""
echo ""

echo "── Privacy Leak Check ──────────────────────"
echo "  Starting server for leak check..."
.venv/Scripts/python -m uvicorn server.app:app --host 0.0.0.0 --port 9999 &
SERVER_PID=$!
sleep 3

RESET_RESPONSE=$(curl -s -X POST http://localhost:9999/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"financial_optimize"}')

if echo "$RESET_RESPONSE" | grep -qi "ground_truth\|outcome_map\|optimal_option"; then
  echo -e "${RED}✗ GROUND TRUTH LEAKED in /reset response — DISQUALIFYING${NC}"
  FAIL=$((FAIL+1))
else
  echo -e "${GREEN}✓ /reset: no ground truth leak${NC}"
  PASS=$((PASS+1))
fi

STEP_RESPONSE=$(curl -s -X POST http://localhost:9999/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"ask_clarification","question_text":"What is the tax bracket?"}')

if echo "$STEP_RESPONSE" | grep -qi "ground_truth\|outcome_map\|optimal_option"; then
  echo -e "${RED}✗ GROUND TRUTH LEAKED in /step response — DISQUALIFYING${NC}"
  FAIL=$((FAIL+1))
else
  echo -e "${GREEN}✓ /step: no ground truth leak${NC}"
  PASS=$((PASS+1))
fi

kill $SERVER_PID 2>/dev/null || true
echo ""

echo "────────────────────────────────────────────"
echo " Results: ${PASS} passed, ${FAIL} failed"
echo "────────────────────────────────────────────"

if [ $FAIL -gt 0 ]; then
  echo -e "${RED}Fix all failures before pushing to HF Space.${NC}"
  exit 1
else
  echo -e "${GREEN}All checks passed. Safe to push.${NC}"
  echo ""
  echo "Next: git add -A && git commit -m 'release' && git push hf main"
  exit 0
fi