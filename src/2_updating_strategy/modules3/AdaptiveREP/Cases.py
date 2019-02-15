"""Define cases to investigate"""

# Parameters
# ----------
# Seed for random number generator
SEED = 10

# Number of calibration intervals
MODEL_HORIZON = 52

# Calibration interval index at which a structural (emissions intensity) shock occurs
SHOCK_INDEX = 10

# Permit price applying for all intervals [$/tCO2]
PERMIT_PRICE = 40

# Number of forecast intervals used by MPC controller
FORECAST_INTERVALS_MPC = 6

# Number of forecast intervals when revenue rebalancing
FORECAST_INTERVALS_REVENUE_REBALANCE = 1

# Used to scale forecast values. Realised (perfect forecast) values from benchmark cases
# are scaled by a uniformly distributed random number in the interval, with interval widening by
# FORECAST_UNCERTAINTY_INCREMENT when moving further into the future.
# E.g. for the first calibration interval the scaling factor will be in (0.95, 1.05), for the second (0.9, 1.1)
FORECAST_UNCERTAINTY_INCREMENT = 0.05

# Scheme revenue during first week
INITIAL_ROLLING_SCHEME_REVENUE = 0

# If ramping scheme revenue, the calibration interval index at which revenue ramp begins
START_REVENUE_RAMP_INDEX = 10

# Number of intervals over which revenue is ramped
REVENUE_RAMP_INTERVALS = 10

# Amount the revenue target is incremented each calibration interval
REVENUE_RAMP_INCREMENT = 3e6

# Cases to investigate
# --------------------
# Benchmark cases (results used to generate forecasts for updating cases)
benchmark_cases = [
    {'description': 'business as usual - no shocks',
     'model_horizon': MODEL_HORIZON,
     'shock_option': 'NO_SHOCKS',
     # 'forecast_shock': False,
     # 'shock_index': SHOCK_INDEX,
     'seed': SEED,
     'update_mode': 'NO_UPDATE',
     'default_baseline': 0,
     'initial_permit_price': 0,
     'initial_rolling_scheme_revenue': INITIAL_ROLLING_SCHEME_REVENUE,
     # 'forecast_intervals': 2,
     # 'forecast_uncertainty_increment': 0.05,
     # 'revenue_target': 'neutral',
     # 'renewables_eligibility': 'ineligible',
     # 'revenue_ramp_calibration_interval_start': 1,
     # 'revenue_ramp_intervals': 10,
     # 'revenue_ramp_increment': 1e6
     },

    {'description': 'business as usual - emissions intensity shock',
     'model_horizon': MODEL_HORIZON,
     'shock_option': 'EMISSIONS_INTENSITY_SHOCK',
     # 'forecast_shock': False,
     'shock_index': SHOCK_INDEX,
     'seed': SEED,
     'update_mode': 'NO_UPDATE',
     'default_baseline': 0,
     'initial_permit_price': 0,
     'initial_rolling_scheme_revenue': INITIAL_ROLLING_SCHEME_REVENUE,
     # 'forecast_intervals': 2,
     # 'forecast_uncertainty_increment': 0.05,
     # 'revenue_target': 'neutral',
     # 'renewables_eligibility': 'ineligible',
     # 'revenue_ramp_calibration_interval_start': 1,
     # 'revenue_ramp_intervals': 10,
     # 'revenue_ramp_increment': 1e6
     },

    {'description': 'carbon tax - no shocks',
     'model_horizon': MODEL_HORIZON,
     'shock_option': 'NO_SHOCKS',
     # 'shock_index': SHOCK_INDEX,
     'seed': SEED,
     'update_mode': 'NO_UPDATE',
     'default_baseline': 0,
     'initial_permit_price': PERMIT_PRICE,
     'initial_rolling_scheme_revenue': INITIAL_ROLLING_SCHEME_REVENUE,
     # 'forecast_intervals': 2,
     # 'forecast_uncertainty_increment': 0.05,
     # 'revenue_target': 'neutral',
     # 'renewables_eligibility': 'ineligible',
     # 'revenue_ramp_calibration_interval_start': 1,
     # 'revenue_ramp_intervals': 10,
     # 'revenue_ramp_increment': 1e6
     },

    {'description': 'carbon tax - emissions intensity shock',
     'model_horizon': MODEL_HORIZON,
     'shock_option': 'EMISSIONS_INTENSITY_SHOCK',
     # 'forecast_shock': False,
     'shock_index': SHOCK_INDEX,
     'seed': SEED,
     'update_mode': 'NO_UPDATE',
     'default_baseline': 0,
     'initial_permit_price': PERMIT_PRICE,
     'initial_rolling_scheme_revenue': INITIAL_ROLLING_SCHEME_REVENUE,
     # 'forecast_intervals': 2,
     # 'forecast_uncertainty_increment': 0.05,
     # 'revenue_target': 'neutral',
     # 'renewables_eligibility': 'ineligible',
     # 'revenue_ramp_calibration_interval_start': 1,
     # 'revenue_ramp_intervals': 10,
     # 'revenue_ramp_increment': 1e6
     },
]

# Updating cases
updating_cases = [
    {'description': 'revenue rebalance update - revenue neutral target - no shocks - renewables ineligible',
     'model_horizon': MODEL_HORIZON,
     'shock_option': 'NO_SHOCKS',
     # 'forecast_shock': False,
     # 'shock_index': SHOCK_INDEX,
     'seed': SEED,
     'update_mode': 'REVENUE_REBALANCE_UPDATE',
     'default_baseline': 1.02,
     'initial_permit_price': PERMIT_PRICE,
     'initial_rolling_scheme_revenue': INITIAL_ROLLING_SCHEME_REVENUE,
     'forecast_intervals': FORECAST_INTERVALS_REVENUE_REBALANCE,
     'forecast_uncertainty_increment': FORECAST_UNCERTAINTY_INCREMENT,
     'revenue_target': 'neutral',
     'renewables_eligibility': 'ineligible',
     # 'revenue_ramp_calibration_interval_start': 1,
     # 'revenue_ramp_intervals': 10,
     # 'revenue_ramp_increment': 1e6
     },

    {'description': 'revenue rebalance update - revenue neutral target - emissions intensity shock - renewables ineligible',
     'model_horizon': MODEL_HORIZON,
     'shock_option': 'EMISSIONS_INTENSITY_SHOCK',
     # 'forecast_shock': False,
     'shock_index': SHOCK_INDEX,
     'seed': SEED,
     'update_mode': 'REVENUE_REBALANCE_UPDATE',
     'default_baseline': 1.02,
     'initial_permit_price': PERMIT_PRICE,
     'initial_rolling_scheme_revenue': INITIAL_ROLLING_SCHEME_REVENUE,
     'forecast_intervals': FORECAST_INTERVALS_REVENUE_REBALANCE,
     'forecast_uncertainty_increment': FORECAST_UNCERTAINTY_INCREMENT,
     'revenue_target': 'neutral',
     'renewables_eligibility': 'ineligible',
     # 'revenue_ramp_calibration_interval_start': 1,
     # 'revenue_ramp_intervals': 10,
     # 'revenue_ramp_increment': 1e6
     },

    {'description': 'mpc update - revenue neutral target - no shocks - renewables ineligible',
     'model_horizon': MODEL_HORIZON,
     'shock_option': 'NO_SHOCKS',
     # 'forecast_shock': False,
     # 'shock_index': SHOCK_INDEX,
     'seed': SEED,
     'update_mode': 'MPC_UPDATE',
     'default_baseline': 1.02,
     'initial_permit_price': PERMIT_PRICE,
     'initial_rolling_scheme_revenue': INITIAL_ROLLING_SCHEME_REVENUE,
     'forecast_intervals': FORECAST_INTERVALS_MPC,
     'forecast_uncertainty_increment': FORECAST_UNCERTAINTY_INCREMENT,
     'revenue_target': 'neutral',
     'renewables_eligibility': 'ineligible',
     # 'revenue_ramp_calibration_interval_start': 1,
     # 'revenue_ramp_intervals': 10,
     # 'revenue_ramp_increment': 1e6
     },

    {'description': 'mpc update - revenue neutral target - emissions intensity shock - renewables ineligible',
     'model_horizon': MODEL_HORIZON,
     'shock_option': 'EMISSIONS_INTENSITY_SHOCK',
     # 'forecast_shock': False,
     'shock_index': SHOCK_INDEX,
     'seed': SEED,
     'update_mode': 'MPC_UPDATE',
     'default_baseline': 1.02,
     'initial_permit_price': PERMIT_PRICE,
     'initial_rolling_scheme_revenue': INITIAL_ROLLING_SCHEME_REVENUE,
     'forecast_intervals': FORECAST_INTERVALS_MPC,
     'forecast_uncertainty_increment': FORECAST_UNCERTAINTY_INCREMENT,
     'revenue_target': 'neutral',
     'renewables_eligibility': 'ineligible',
     # 'revenue_ramp_calibration_interval_start': 1,
     # 'revenue_ramp_intervals': 10,
     # 'revenue_ramp_increment': 1e6
     },

    {'description': 'revenue rebalance update - revenue ramp up target - no shocks - renewables ineligible',
     'model_horizon': MODEL_HORIZON,
     'shock_option': 'NO_SHOCKS',
     # 'forecast_shock': False,
     # 'shock_index': SHOCK_INDEX,
     'seed': SEED,
     'update_mode': 'REVENUE_REBALANCE_UPDATE',
     'default_baseline': 1.02,
     'initial_permit_price': PERMIT_PRICE,
     'initial_rolling_scheme_revenue': INITIAL_ROLLING_SCHEME_REVENUE,
     'forecast_intervals': FORECAST_INTERVALS_REVENUE_REBALANCE,
     'forecast_uncertainty_increment': FORECAST_UNCERTAINTY_INCREMENT,
     'revenue_target': 'ramp_up',
     'renewables_eligibility': 'ineligible',
     'revenue_ramp_calibration_interval_start': START_REVENUE_RAMP_INDEX,
     'revenue_ramp_intervals': REVENUE_RAMP_INTERVALS,
     'revenue_ramp_increment': REVENUE_RAMP_INCREMENT,
     },

    {'description': 'mpc update - revenue ramp up target - no shocks - renewables ineligible',
     'model_horizon': MODEL_HORIZON,
     'shock_option': 'NO_SHOCKS',
     # 'forecast_shock': False,
     # 'shock_index': SHOCK_INDEX,
     'seed': SEED,
     'update_mode': 'MPC_UPDATE',
     'default_baseline': 1.02,
     'initial_permit_price': PERMIT_PRICE,
     'initial_rolling_scheme_revenue': INITIAL_ROLLING_SCHEME_REVENUE,
     'forecast_intervals': FORECAST_INTERVALS_MPC,
     'forecast_uncertainty_increment': FORECAST_UNCERTAINTY_INCREMENT,
     'revenue_target': 'ramp_up',
     'renewables_eligibility': 'ineligible',
     'revenue_ramp_calibration_interval_start': START_REVENUE_RAMP_INDEX,
     'revenue_ramp_intervals': REVENUE_RAMP_INTERVALS,
     'revenue_ramp_increment': REVENUE_RAMP_INCREMENT,
     },

    {'description': 'revenue rebalance update - revenue neutral target - emissions intensity shock unanticipated - renewables ineligible',
     'model_horizon': MODEL_HORIZON,
     'shock_option': 'EMISSIONS_INTENSITY_SHOCK',
     'forecast_shock': True,
     'shock_index': SHOCK_INDEX,
     'seed': SEED,
     'update_mode': 'REVENUE_REBALANCE_UPDATE',
     'default_baseline': 1.02,
     'initial_permit_price': PERMIT_PRICE,
     'initial_rolling_scheme_revenue': INITIAL_ROLLING_SCHEME_REVENUE,
     'forecast_intervals': FORECAST_INTERVALS_REVENUE_REBALANCE,
     'forecast_uncertainty_increment': FORECAST_UNCERTAINTY_INCREMENT,
     'revenue_target': 'neutral',
     'renewables_eligibility': 'ineligible',
     'revenue_ramp_calibration_interval_start': START_REVENUE_RAMP_INDEX,
     'revenue_ramp_intervals': REVENUE_RAMP_INTERVALS,
     'revenue_ramp_increment': REVENUE_RAMP_INCREMENT,
     },

    {'description': 'mpc update - revenue neutral target - emissions intensity shock unanticipated - renewables ineligible',
     'model_horizon': MODEL_HORIZON,
     'shock_option': 'EMISSIONS_INTENSITY_SHOCK',
     'forecast_shock': True,
     'shock_index': SHOCK_INDEX,
     'seed': SEED,
     'update_mode': 'MPC_UPDATE',
     'default_baseline': 1.02,
     'initial_permit_price': PERMIT_PRICE,
     'initial_rolling_scheme_revenue': INITIAL_ROLLING_SCHEME_REVENUE,
     'forecast_intervals': FORECAST_INTERVALS_MPC,
     'forecast_uncertainty_increment': FORECAST_UNCERTAINTY_INCREMENT,
     'revenue_target': 'neutral',
     'renewables_eligibility': 'ineligible',
     'revenue_ramp_calibration_interval_start': START_REVENUE_RAMP_INDEX,
     'revenue_ramp_intervals': REVENUE_RAMP_INTERVALS,
     'revenue_ramp_increment': REVENUE_RAMP_INCREMENT,
     },

]
