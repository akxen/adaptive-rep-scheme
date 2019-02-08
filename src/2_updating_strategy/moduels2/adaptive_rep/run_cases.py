# Paths
# -----
# Directory containing model output
results_dir = os.path.join(os.path.curdir, 'output')

# Directory containing network and generator data
data_dir = os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, os.path.pardir, 'data')

# Path to scenarios directory
scenarios_dir = os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, '1_create_scenarios', 'output')


# Default parameters
# ------------------
# Total number of week considered by model (starting from week 1)
model_horizon = 52

# Week at which emissions intensity shock occurs (if specified to happen)
week_of_shock = 10

# Seed used for random number generator
seed = 10

# Default permit price to be used for simulations
initial_permit_price = 40

# Number of forecast intervals for revenue re-balancing rule
forecast_intervals_revenue_rebalance = 1

# Number of forecast intervals for model predictive control (MPC) updating rule
forecast_intervals_mpc = 6

# Default baseline. (Will be used as baseline in the interval preceding the
# first forecast interval when implementing MPC updates)
default_baseline = 1.0228

# Rolling scheme revenue at scheme start
initial_rolling_scheme_revenue = 0

# Amount by which to increment scaling factor each week in future when consider
# imperfect forecasts
forecast_uncertainty_increment = 0.05


# Revenue neutral target
# ----------------------
revenue_neutral_target = {i: 0 for i in range(1, model_horizon + 1)}


# Positive scheme revenue target
# ------------------------------
# Initialise positive revenue target dictionary
positive_revenue_target = dict()

# Week when revenue ramp is to begin. Set to week_of shock
revenue_ramp_week_start = week_of_shock

# Number of intervals over which revenue target is ramped up
revenue_ramp_intervals = 10

# Amount to increment revenue target each period when ramping revenue
revenue_ramp_increment = 3e6

# Ramp revenue by increasing increment after week_of_shock for predefined number of weeks.
# Then maintain revenue at specified level.
for key, value in revenue_neutral_target.items():

    # Mainitain intial scheme revenue if before week_of_shock
    if key < week_of_shock:
        positive_revenue_target[key] = initial_rolling_scheme_revenue

    # Ramp scheme revenue by the same increment following for
    elif (key >= week_of_shock) and (key < week_of_shock + revenue_ramp_intervals):
        positive_revenue_target[key] = positive_revenue_target[key - 1] + revenue_ramp_increment

    # Maintain rolling scheme revenue at new level
    else:
        positive_revenue_target[key] = positive_revenue_target[key - 1]


# Renewables eligibility
# ----------------------
# Dictionary to use if renewables ineligible for all periods
renewables_ineligible = {i: False for i in range(1, model_horizon + 1)}

# Week in which renewables become eligible for payments (if specified).
# Set to week_of_shock
renewables_eligible_week_start = week_of_shock

# Dictionary to use if renewables are eligible to receive payments
renewables_become_eligible = {i: False if i < renewables_eligible_week_start else True for i in range(1, model_horizon + 1)}


for case, options in cases.items():
    # Run case
    print(case)
    run_id = run_case(data_dir=data_dir,
                      scenarios_dir=scenarios_dir,
                      run_summaries=run_summaries,
                      shock_option=options['shock_option'],
                      update_mode=options['update_mode'],
                      initial_permit_price=initial_permit_price,
                      model_horizon=model_horizon,
                      forecast_intervals=options['forecast_intervals'],
                      forecast_uncertainty_increment=options['forecast_uncertainty_increment'],
                      weekly_target_scheme_revenue=options['weekly_target_scheme_revenue'],
                      renewables_eligibility=options['renewables_eligibility'],
                      description=case,
                      week_of_shock=options['week_of_shock'],
                      forecast_shock=options['forecast_shock'],
                      default_baseline=default_baseline,
                      initial_rolling_scheme_revenue=initial_rolling_scheme_revenue,
                      seed=seed)

    # Check that updating rule has worked correctly
    with open(f'output/{run_id}_week_metrics.pickle', 'rb') as f:
        week_metrics = pickle.load(f)

    print(pd.DataFrame(week_metrics)[['baseline',
                                      'rolling_scheme_revenue_interval_end',
                                      'net_scheme_revenue_dispatchable_generators',
                                      'net_scheme_revenue_intermittent_generators']])
