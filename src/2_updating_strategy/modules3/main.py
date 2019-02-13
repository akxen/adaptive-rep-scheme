"""Run agent-based model for different cases"""

import os

from AdaptiveREP import Simulator, Cases

# Paths
# -----
# Core data directory
data_dir = os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, os.path.pardir, 'data')

# Directory containing representative scenario data for each calibration interval (week)
scenarios_dir = os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, '1_create_scenarios', 'output')

# Directory for output files
output_dir = os.path.join(os.path.curdir, 'output')


# Run model
# ---------
# Benchmark cases
for case_options in Cases.benchmark_cases:
    # Object used to run agent-based simulation
    Sim = Simulator.RunSimulator(data_dir=data_dir, scenarios_dir=scenarios_dir, output_dir=output_dir, **case_options)

    # Unique ID corresponding to case
    case_id = Sim.run_case()

    print(f'Finished case: {case_id}')

# Baseline updating cases
for case_options in Cases.updating_cases:
    # Object used to run agent-based simulation
    Sim = Simulator.RunSimulator(data_dir=data_dir, scenarios_dir=scenarios_dir, output_dir=output_dir, **case_options)

    # Unique ID corresponding to case
    case_id = Sim.run_case()

    print(f'Finished case: {case_id}')
