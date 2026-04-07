from flidie_environment import FlidieEnv
from models import FinancialAction

env = FlidieEnv()

obs = env.reset("financial_optimize")
print("Scenario:", obs.scenario_id)
print("Title:", obs.title[:50])

r1 = env.step(FinancialAction(
    action_type="calculate",
    expression="90000*0.3",
    expected_result=27000
))
print("After calculate — reward:", r1.reward)
print("calculations_done:", r1.observation.calculations_done)

r2 = env.step(FinancialAction(
    action_type="choose_option",
    option_id="A"
))
print("Final reward:", r2.reward)
print("Done:", r2.done)