import os

from AdaptiveREP import Simulator

data_dir = os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, os.path.pardir, 'data')
output_dir = os.path.join(os.path.curdir, 'output')
scenarios_dir = os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, '1_create_scenarios', 'output')


case = {'description': 'business as usual - no shocks',
        'shock_option': 'NO_SHOCKS',
        'update_mode': 'NO_UPDATE',
        'default_baseline': 0,
        'initial_permit_price': 0,
        'initial_rolling_scheme_revenue': 0,
        'model_horizon': 2}

Simulator.run_case(data_dir=data_dir, scenarios_dir=scenarios_dir, output_dir=output_dir, **case)
