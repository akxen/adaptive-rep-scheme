"""Classes used to run agent-based simulations"""

import time

from .MPC import MPCModel
from .DCOPF import DCOPFModel
from .Targets import RevenueTarget
from .Baseline import Updater
from .Forecast import ConstructForecast
from .ModelUtils import RecordResults, Utils, ApplyShock
from .Eligibility import RenewablesEligibility


class RunSimulator:
  """Class used to run agent-based simulations"""

  def __init__(self, data_dir, scenarios_dir, output_dir, **case_options):
    """Run model with given parameters

    Parameters
    ----------
    data_dir : str
        Path to directory containing core data files used in model initialisation

    scenarios_dir : str
        Path to directory containing representative operating scenarios

    output_dir : str
        Path to directory where results files will be written


    **case_options : dict
        Case parameters
          'description' - description of model being run
          'model_horizon' - number of calibration intervals in model horizon
          'shock_option' - type of shock to investigate (options: 'NO_SHOCKS', 'EMISSIONS_INTENSITY_SHOCK')
          'forecast_shock' - Does a forecast shock occur (options: True = shock is unforseen, False = shock is foreseen)
          'shock_index' - index at which shock occurs
          'seed' - seed for random number generator (make results reproducible)
          'update_mode' - type of baseline updating mode (options: 'NO_UPDATE', 'REVENUE_REBALANCE_UPDATE', 'MPC_UPDATE'
          'default_baseline' - emissions intensity baseline default value [tCO2/MWh]
          'initial_permit_price' - emission permit price [$/tCO2]
          'initial_rolling_scheme_revenue' - initial scheme revenue
          'forecast_intervals' - number of calibration intervals in forecast horizon
          'forecast_uncertainty_increment' - increment by which forecast values are perturbed each forecast interval
          'revenue_target' - type of revenue target (options: 'neutral' - want zero net revenue, 'ramp_up' - want positive net revenue)
          'renewables_eligibility' - Indicator if renewables eligible for emissions payements (options: 'ineligible', 'eligible')
          'revenue_ramp_calibration_interval_start' - calibration interval index at which scheme revenue target begins to ramp up
          'revenue_ramp_intervals' - number of calibration intervals the revenue target is increased
          'revenue_ramp_increment' - amount the revenue target is incremented each calibration interval when ramping up.

    Returns
    -------
    case_id : str
        ID used to identify case investigated
    """

    # Path to directory containing core data files
    self.data_dir = data_dir

    # Path to directory containing representative operating scenarios
    self.scenarios_dir = scenarios_dir

    # Path to directory where model output files will be saved
    self.output_dir = output_dir

    # Case parameters
    self.case_options = case_options

  def run_case(self):
    """Run model for specified case"""

    # Print case being run
    print(f'Running case: {self.case_options}')

    # Check if case options valid
    Utils(case_options=self.case_options).case_options_valid()

    # Create model objects
    # --------------------
    # Instantiate DCOPF model object
    DCOPF = DCOPFModel(data_dir=self.data_dir, scenarios_dir=self.scenarios_dir)

    # Instantiate Model Predictive Controller if MPC update specified
    if self.case_options.get('update_mode') == 'MPC_UPDATE':
      MPC = MPCModel(generator_index=DCOPF.model.OMEGA_G, forecast_intervals=self.case_options.get('forecast_intervals'))

    # Object used to update baseline
    Baseline = Updater()

    # Only generate forecasts and targets if updating cases are investigated
    if self.case_options.get('update_mode') != 'NO_UPDATE':
      # Object used to generate forecasts
      Forecasts = ConstructForecast(output_dir=self.output_dir, case_options=self.case_options)

      # Revenue Target
      Target = RevenueTarget(case_options=self.case_options)

      # Add revenue target dictionary to case options
      self.case_options['target_scheme_revenue'] = Target.revenue_target

      # Defines if renewables are eligible for payments under scheme
      Eligibility = RenewablesEligibility(case_options=self.case_options)

    # If shocks are to be considered
    if self.case_options.get('shock_option') != 'NO_SHOCKS':
      # Object used to generate emissions intensity shock
      Shock = ApplyShock(output_dir=self.output_dir, case_options=self.case_options)

    # Object used to record model results and compute calibration interval metrics
    Results = RecordResults(case_options=self.case_options)

    # Run scenarios
    # -------------
    # Calibration intervals for which model will be run (can run for less than one year by adjusting model_horizon)
    # Note: weekly calibration intervals are used in this analysis
    calibration_intervals = range(1, self.case_options.get('model_horizon') + 1)

    # For each calibration interval
    for calibration_interval in calibration_intervals:
      # Start clock to see how long it takes to solve all scenarios for each calibration interval
      t0 = time.time()

      # If the first calibration interval, initialise policy parameters
      if calibration_interval == calibration_intervals[0]:
        # Initialise permit price
        permit_price = self.case_options.get('initial_permit_price')

        # Initialise rolling scheme revenue (value at the end of previous calibration interval)
        Results.calibration_interval_metrics['rolling_scheme_revenue_interval_start'][calibration_interval] = self.case_options.get('initial_rolling_scheme_revenue')

      # Update rolling scheme revenue (total amount of scheme revenue at start of interval)
      else:
        Results.calibration_interval_metrics['rolling_scheme_revenue_interval_start'][calibration_interval] = Results.calibration_interval_metrics['rolling_scheme_revenue_interval_end'][calibration_interval - 1]

      # Compute baseline
      # ----------------
      # If no updates are specified, keep baseline at same (default) level
      if self.case_options.get('update_mode') == 'NO_UPDATE':
        baseline = self.case_options.get('default_baseline')

      # If the revenue rebalancing algorithm is specified
      elif self.case_options.get('update_mode') == 'REVENUE_REBALANCE_UPDATE':
        # Resolve current revenue imbalance by adjust baseline that will apply in the coming calibration interval
        baseline = Baseline.get_revenue_rebalance_update(model_object=DCOPF,
                                                         case_options=self.case_options,
                                                         calibration_interval=calibration_interval,
                                                         forecast_generator_energy=Forecasts.generator_energy,
                                                         forecast_emissions_intensities=Forecasts.generator_emissions_intensities,
                                                         forecast_intermittent_energy=Forecasts.intermittent_energy,
                                                         renewables_eligibility=Eligibility.renewables_eligibility[calibration_interval],
                                                         permit_price=self.case_options.get('initial_permit_price'),
                                                         scheme_revenue_interval_start=Results.calibration_interval_metrics['rolling_scheme_revenue_interval_start'][calibration_interval],
                                                         target_scheme_revenue=Target.revenue_target,
                                                         )

      # If the Model Predictive Control update is specified
      elif self.case_options.get('update_mode') == 'MPC_UPDATE':
        # If first calibration interval, initialise baseline. This is an assumed value that applied
        # in the preceding calibration interval (prior to model start)
        if calibration_interval == 1:
          # Set baseline to the default value
          baseline = self.case_options.get('default_baseline')

        # Use Model Predictive Controller to update baseline
        baseline = Baseline.get_mpc_update(model_object=MPC,
                                           case_options=self.case_options,
                                           calibration_interval=calibration_interval,
                                           forecast_generator_energy=Forecasts.generator_energy,
                                           forecast_emissions_intensities=Forecasts.generator_emissions_intensities,
                                           forecast_intermittent_energy=Forecasts.intermittent_energy,
                                           renewables_eligibility=Eligibility.renewables_eligibility,
                                           permit_price=self.case_options.get('initial_permit_price'),
                                           baseline_interval_start=baseline,
                                           scheme_revenue_interval_start=Results.calibration_interval_metrics['rolling_scheme_revenue_interval_start'][calibration_interval],
                                           target_scheme_revenue=Target.revenue_target)

      # Record baseline applying for the given calibration interval
      Results.calibration_interval_metrics['baseline'][calibration_interval] = baseline

      # Apply emissions intensity shock for given calibration interval if shock option specified
      if ((self.case_options.get('shock_option') == 'EMISSIONS_INTENSITY_SHOCK') and
              (calibration_interval == self.case_options.get('shock_index'))):
        # Update emissions intensities in DCOPF model object
        DCOPF = Shock.apply_emissions_intensity_shock(model_object=DCOPF)

      # For each representative scenario approximating a calibration interval's operating state
      for scenario_index in DCOPF.df_scenarios.columns.levels[1]:
        # Update model parameters
        DCOPF.update_model_parameters(calibration_interval=calibration_interval,
                                      scenario_index=scenario_index,
                                      baseline=baseline,
                                      permit_price=permit_price)

        # Solve model
        DCOPF.solve_model()

        # Store scenario results
        Results.store_scenario_results(model_object=DCOPF,
                                       calibration_interval=calibration_interval,
                                       scenario_index=scenario_index)

      # Compute calibration interval metrics
      Results.store_calibration_interval_metrics(model_object=DCOPF,
                                                 calibration_interval=calibration_interval,
                                                 case_options=self.case_options)

      print(f'Completed calibration interval {calibration_interval} in {time.time()-t0:.2f}s')

    # Save results
    # ------------
    # Save calibration interval results
    Results.save_results(output_dir=self.output_dir, model_object=DCOPF)

    return Results.case_summary['case_id']
