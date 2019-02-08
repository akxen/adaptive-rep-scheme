from .model_utils import get_perfect_forecasts, get_perturbed_forecast, get_formatted_weekly_input

SEED = 10

MODEL_HORIZON = 5

WEEK_OF_SHOCK = 2

INITIAL_PERMIT_PRICE = 40

FORECAST_INTERVALS_MPC = 6

FORECAST_INTERVALS_REVENUE_REBALANCE = 1

FORECAST_UNCERTAINTY_INCREMENT = 0.05

INITIAL_ROLLING_SCHEME_REVENUE = 0


benchmark_cases = [{'description': 'business as usual - no shocks',
                    'shock_option': 'NO_SHOCKS',
                    'update_mode': 'NO_UPDATE',
                    'default_baseline': 0,
                    'initial_permit_price': 0,
                    'initial_rolling_scheme_revenue': 0,
                    'model_horizon': MODEL_HORIZON},

                   {'description': 'business as usual - emissions intensity shock',
                    'shock_option': 'EMISSIONS_INTENSITY_SHOCK',
                    'update_mode': 'NO_UPDATE',
                    'default_baseline': 0,
                    'initial_permit_price': 0,
                    'initial_rolling_scheme_revenue': 0,
                    'seed': SEED,
                    'model_horizon': MODEL_HORIZON,
                    'week_of_shock': WEEK_OF_SHOCK},

                   {'description': 'carbon tax - no shocks',
                    'shock_option': 'NO_SHOCKS',
                    'update_mode': 'NO_UPDATE',
                    'default_baseline': 0,
                    'initial_permit_price': 40,
                    'initial_rolling_scheme_revenue': 0,
                    'model_horizon': MODEL_HORIZON},

                   {'description': 'carbon tax - emissions intensity shock',
                    'shock_option': 'EMISSIONS_INTENSITY_SHOCK',
                    'update_mode': 'NO_UPDATE',
                    'default_baseline': 0,
                    'initial_permit_price': 40,
                    'initial_rolling_scheme_revenue': 0,
                    'seed': SEED,
                    'model_horizon': MODEL_HORIZON,
                    'week_of_shock': WEEK_OF_SHOCK},
                   ]


# Revenue neutral target
# ----------------------
revenue_neutral_target = {i: 0 for i in range(1, MODEL_HORIZON + 1)}


# Positive scheme revenue target
# ------------------------------
# Initialise positive revenue target dictionary
positive_revenue_target = dict()

# Week when revenue ramp is to begin. Set to week_of shock
revenue_ramp_week_start = WEEK_OF_SHOCK

# Number of intervals over which revenue target is ramped up
revenue_ramp_intervals = 10

# Amount to increment revenue target each period when ramping revenue
revenue_ramp_increment = 3e6

# Ramp revenue by increasing increment after week_of_shock for predefined number of weeks.
# Then maintain revenue at specified level.
for key, value in revenue_neutral_target.items():

  # Mainitain intial scheme revenue if before week_of_shock
  if key < WEEK_OF_SHOCK:
    positive_revenue_target[key] = INITIAL_ROLLING_SCHEME_REVENUE

  # Ramp scheme revenue by the same increment following for
  elif (key >= WEEK_OF_SHOCK) and (key < WEEK_OF_SHOCK + revenue_ramp_intervals):
    positive_revenue_target[key] = positive_revenue_target[key - 1] + revenue_ramp_increment

  # Maintain rolling scheme revenue at new level
  else:
    positive_revenue_target[key] = positive_revenue_target[key - 1]


# Renewables eligibility
# ----------------------
# Dictionary to use if renewables ineligible for all periods
renewables_ineligible = {i: False for i in range(1, MODEL_HORIZON + 1)}

# Week in which renewables become eligible for payments (if specified).
# Set to week_of_shock
renewables_eligible_week_start = WEEK_OF_SHOCK

# Dictionary to use if renewables are eligible to receive payments
renewables_become_eligible = {i: False if i < renewables_eligible_week_start else True for i in range(1, MODEL_HORIZON + 1)}


updating_cases = [
    # Perfect forecasts - no shocks
    {'description': 'Revenue rebalance update - positive_revenue_target - no shocks - perfect forecast',
     'shock_option': 'NO_SHOCKS',
     'update_mode': 'REVENUE_REBALANCE_UPDATE',
     'forecast_intervals': FORECAST_INTERVALS_REVENUE_REBALANCE,
     'forecast_uncertainty_increment': 0,
     'weekly_target_scheme_revenue': positive_revenue_target,
     'renewables_eligibility': renewables_ineligible,
     'forecast_shock': False},

    {'description': 'MPC update - revenue neutral - no shocks - perfect forecast',
     'shock_option': 'NO_SHOCKS',
     'update_mode': 'MPC_UPDATE',
     'week_of_shock': None,
     'forecast_intervals': FORECAST_INTERVALS_MPC,
     'forecast_uncertainty_increment': 0,
     'weekly_target_scheme_revenue': positive_revenue_target,
     'renewables_eligibility': renewables_ineligible,
     'forecast_shock': False},

    # Perfect forecast - emissions intensity shock
    {'description': 'Revenue rebalance update - positive_revenue_target - emissions intensity shock - perfect forecast',
     'shock_option': 'EMISSIONS_INTENSITY_SHOCK',
     'update_mode': 'REVENUE_REBALANCE_UPDATE',
     'week_of_shock': WEEK_OF_SHOCK,
     'forecast_intervals': FORECAST_INTERVALS_REVENUE_REBALANCE,
     'forecast_uncertainty_increment': 0,
     'weekly_target_scheme_revenue': positive_revenue_target,
     'renewables_eligibility': renewables_ineligible,
     'forecast_shock': False},

    {'description': 'MPC update - positive_revenue_target - emissions intensity shock - perfect forecast',
     'shock_option': 'EMISSIONS_INTENSITY_SHOCK',
     'update_mode': 'MPC_UPDATE',
     'week_of_shock': WEEK_OF_SHOCK,
     'forecast_intervals': FORECAST_INTERVALS_MPC,
     'forecast_uncertainty_increment': 0,
     'weekly_target_scheme_revenue': positive_revenue_target,
     'renewables_eligibility': renewables_ineligible,
     'forecast_shock': False},

    # Revenue neutral
    {'description': 'Revenue rebalance update - revenue neutral - no shocks - imperfect forecast',
     'shock_option': 'NO_SHOCKS',
     'update_mode': 'REVENUE_REBALANCE_UPDATE',
     'week_of_shock': None,
     'forecast_intervals': FORECAST_INTERVALS_REVENUE_REBALANCE,
     'forecast_uncertainty_increment': FORECAST_UNCERTAINTY_INCREMENT,
     'weekly_target_scheme_revenue': revenue_neutral_target,
     'renewables_eligibility': renewables_ineligible,
     'forecast_shock': False},

    {'description': 'MPC update - revenue neutral - no shocks - imperfect forecast',
     'shock_option': 'NO_SHOCKS',
     'update_mode': 'MPC_UPDATE',
     'week_of_shock': None,
     'forecast_intervals': FORECAST_INTERVALS_MPC,
     'forecast_uncertainty_increment': FORECAST_UNCERTAINTY_INCREMENT,
     'weekly_target_scheme_revenue': revenue_neutral_target,
     'renewables_eligibility': renewables_ineligible,
     'forecast_shock': False},

    # Positive scheme revenue target
    {'description': 'Revenue rebalance update - positive revenue target - no shocks - imperfect forecast',
     'shock_option': 'NO_SHOCKS',
     'update_mode': 'REVENUE_REBALANCE_UPDATE',
     'week_of_shock': None,
     'forecast_intervals': FORECAST_INTERVALS_REVENUE_REBALANCE,
     'forecast_uncertainty_increment': FORECAST_UNCERTAINTY_INCREMENT,
     'weekly_target_scheme_revenue': positive_revenue_target,
     'renewables_eligibility': renewables_ineligible,
     'forecast_shock': False},

    {'description': 'MPC update - positive revenue target - no shocks - imperfect forecast',
     'shock_option': 'NO_SHOCKS',
     'update_mode': 'MPC_UPDATE',
     'week_of_shock': None,
     'forecast_intervals': FORECAST_INTERVALS_MPC,
     'forecast_uncertainty_increment': FORECAST_UNCERTAINTY_INCREMENT,
     'weekly_target_scheme_revenue': positive_revenue_target,
     'renewables_eligibility': renewables_ineligible,
     'forecast_shock': False},

    # Anticipated emissions intensity shock
    {'description': 'Revenue rebalance update - revenue neutral - anticipated emissions intensity shock - imperfect forecast',
     'shock_option': 'EMISSIONS_INTENSITY_SHOCK',
     'update_mode': 'REVENUE_REBALANCE_UPDATE',
     'week_of_shock': WEEK_OF_SHOCK,
     'forecast_intervals': FORECAST_INTERVALS_REVENUE_REBALANCE,
     'forecast_uncertainty_increment': FORECAST_UNCERTAINTY_INCREMENT,
     'weekly_target_scheme_revenue': revenue_neutral_target,
     'renewables_eligibility': renewables_ineligible,
     'forecast_shock': False},

    {'description': 'MPC update - revenue neutral - anticipated emissions intensity shock - imperfect forecast',
     'shock_option': 'EMISSIONS_INTENSITY_SHOCK',
     'update_mode': 'MPC_UPDATE',
     'week_of_shock': WEEK_OF_SHOCK,
     'forecast_intervals': FORECAST_INTERVALS_MPC,
     'forecast_uncertainty_increment': FORECAST_UNCERTAINTY_INCREMENT,
     'weekly_target_scheme_revenue': revenue_neutral_target,
     'renewables_eligibility': renewables_ineligible,
     'forecast_shock': False},

    # Unanticipated emissions intensity shock
    {'description': 'Revenue rebalance update - revenue neutral - unanticipated emissions intensity shock - imperfect forecast',
     'shock_option': 'EMISSIONS_INTENSITY_SHOCK',
     'update_mode': 'REVENUE_REBALANCE_UPDATE',
     'week_of_shock': WEEK_OF_SHOCK,
     'forecast_intervals': FORECAST_INTERVALS_REVENUE_REBALANCE,
     'forecast_uncertainty_increment': FORECAST_UNCERTAINTY_INCREMENT,
     'weekly_target_scheme_revenue': revenue_neutral_target,
     'renewables_eligibility': renewables_ineligible,
     'forecast_shock': True},

    {'description': 'MPC update - revenue neutral - unanticipated emissions intensity shock - imperfect forecast',
     'shock_option': 'EMISSIONS_INTENSITY_SHOCK',
     'update_mode': 'MPC_UPDATE',
     'week_of_shock': WEEK_OF_SHOCK,
     'forecast_intervals': FORECAST_INTERVALS_MPC,
     'forecast_uncertainty_increment': FORECAST_UNCERTAINTY_INCREMENT,
     'weekly_target_scheme_revenue': revenue_neutral_target,
     'renewables_eligibility': renewables_ineligible,
     'forecast_shock': True},

    # Renewables become eligible for payments under scheme
    {'description': 'Revenue rebalance update - revenue neutral - intermittent renewables become eligible - imperfect forecast',
     'shock_option': 'NO_SHOCKS',
     'update_mode': 'REVENUE_REBALANCE_UPDATE',
     'week_of_shock': WEEK_OF_SHOCK,
     'forecast_intervals': FORECAST_INTERVALS_REVENUE_REBALANCE,
     'forecast_uncertainty_increment': FORECAST_UNCERTAINTY_INCREMENT,
     'weekly_target_scheme_revenue': revenue_neutral_target,
     'renewables_eligibility': renewables_become_eligible,
     'forecast_shock': False},

    {'description': 'MPC update - revenue neutral - intermittent renewables become eligible - imperfect forecast',
     'shock_option': 'NO_SHOCKS',
     'update_mode': 'MPC_UPDATE',
     'week_of_shock': WEEK_OF_SHOCK,
     'forecast_intervals': FORECAST_INTERVALS_MPC,
     'forecast_uncertainty_increment': FORECAST_UNCERTAINTY_INCREMENT,
     'weekly_target_scheme_revenue': revenue_neutral_target,
     'renewables_eligibility': renewables_become_eligible,
     'forecast_shock': False},
]
