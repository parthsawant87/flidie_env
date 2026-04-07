import requests

BASE = "http://localhost:8000"

# Reset
r = requests.post(f"{BASE}/reset", json={"task_id": "financial_optimize"})
print("Reset:", r.status_code)
print("Scenario:", r.json().get("scenario_id"))

# Calculate
r = requests.post(f"{BASE}/step", json={
    "action_type": "calculate",
    "expression": "90000*0.3",
    "expected_result": 27000
})
print("Calculate reward:", r.json().get("reward"))

# Decide
r = requests.post(f"{BASE}/step", json={
    "action_type": "choose_option",
    "option_id": "A"
})
print("Final reward:", r.json().get("reward"))
print("Done:", r.json().get("done"))