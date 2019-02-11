import os

from AdaptiveREP import RunScenarios, Cases


# Parameters
# ----------
# Core data directory containing generator and network datasets
DATA_DIR = r'C:\Users\eee\Desktop\git\rep-updating-strategy\data'

# Representative operating scenarios
SCENARIOS_DIR = r'C:\Users\eee\Desktop\git\rep-updating-strategy\src\1_create_scenarios\output'

# Ouptut director
OUTPUT_DIR = os.path.join(os.path.curdir, 'output')

# Run model for benchmark cases (no updates to emissions intensity baseline)
# for case in cases.benchmark_cases:
#     run_scenarios.run_scenarios(data_dir=DATA_DIR, scenarios_dir=SCENARIOS_DIR, output_dir=OUTPUT_DIR, **case)

# Run model where baseline is updated using different strategies / scenarios
for case in Cases.updating_cases:
    RunScenarios.run_scenarios(data_dir=DATA_DIR, scenarios_dir=SCENARIOS_DIR, output_dir=OUTPUT_DIR, **case)
