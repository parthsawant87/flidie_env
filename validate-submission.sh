BASE=${1:-"http://localhost:8000"}
PASS=0; FAIL=0

check() {
    if eval "$2" > /dev/null 2>&1; then
        echo "  ✓ $1"; PASS=$((PASS+1))
    else
        echo "  ✗ $1"; FAIL=$((FAIL+1))
    fi
}

echo "FLIDIE Submission Validator — $BASE"
echo "═══════════════════════════════════════"

echo ""
echo "[ Structure ]"
check "inference.py at root"         "test -f inference.py"
check "Dockerfile at root"           "test -f Dockerfile"
check "openenv.yaml at root"         "test -f openenv.yaml"
check "models.py at root"            "test -f models.py"
check "graders.py at root"           "test -f graders.py"
check "tasks.py at root"             "test -f tasks.py"
check "flidie_environment.py at root" "test -f flidie_environment.py"
check "server/app.py exists"         "test -f server/app.py"
check "data/scenarios_easy.json"     "test -f data/scenarios_easy.json"
check "data/scenarios_medium.json"   "test -f data/scenarios_medium.json"
check "data/scenarios_hard.json"     "test -f data/scenarios_hard.json"

echo ""
echo "[ Python imports ]"
check "models.py imports cleanly"    "python3 -c 'from models import FinancialAction, OutcomeTier'"
check "graders.py imports cleanly"   "python3 -c 'from graders import grade_easy, grade_medium, grade_hard'"
check "tasks.py imports cleanly"     "python3 -c 'from tasks import get_task, list_tasks, TASK_REGISTRY'"

echo ""
echo "[ Live endpoints ]"
check "GET /health returns 200"      "curl -sf $BASE/health | grep -q 'ok'"
check "/tasks has 3 tasks"           "curl -sf $BASE/tasks | python3 -c \"import sys,json; d=json.load(sys.stdin); assert len(d['tasks'])==3\""

echo ""
echo "[ Security scan ]"
RESET_BODY=$(curl -sf -X POST $BASE/reset -H "Content-Type: application/json" -d '{"task_id":"startup_compliance"}')
check "/reset has no ground_truth"   "echo '$RESET_BODY' | grep -vq 'ground_truth'"
check "/reset has no outcome_map"    "echo '$RESET_BODY' | grep -vq 'outcome_map'"
check "/reset has no optimal_option" "echo '$RESET_BODY' | grep -vq 'optimal_option'"

SID=$(curl -si -X POST $BASE/reset -H "Content-Type: application/json" -d '{"task_id":"financial_optimize"}' | grep -i x-session-id | tr -d '\r' | awk '{print $2}')
STEP_BODY=$(curl -sf -X POST $BASE/step -H "Content-Type: application/json" -H "X-Session-ID: $SID" -d '{"action_type":"ask_clarification","question_text":"What is the 80C limit?"}')
check "/step has no ground_truth"    "echo '$STEP_BODY' | grep -vq 'ground_truth'"

echo ""
echo "[ Test suites ]"
check "test_graders.py all pass"     "python3 -m pytest test_graders.py -q 2>&1 | grep -q '20 passed'"

echo ""
echo "═══════════════════════════════════════"
echo "Results: $PASS passed, $FAIL failed"
if [ $FAIL -eq 0 ]; then
    echo "✓ Ready to push. Run: openenv push --repo-id YOUR_USERNAME/flidie-env"
else
    echo "✗ Fix failures before pushing."
    exit 1
fi
