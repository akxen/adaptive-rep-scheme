import time

from .MPC import MPCModel
from .DCOPF import DCOPFModel
from .Targets import RevenueTarget
from .Baseline import Updater
from .Forecast import ConstructForecast
from .ModelUtils import RecordResults, Utils, ApplyShock
from .Eligibility import RenewablesEligibility


class RunSimulator:
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
        Case options

    Returns
    -------
    case_id : str
        ID used to identify model run
    """
    self.data_dir = data_dir
    self.scenarios_dir = scenarios_dir
    self.output_dir = output_dir
    self.case_options = case_options

  def run_case(self):
    "Run model for specified case"

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

    # Object used to record model results and compute calibration interval metrics
    Results = RecordResults(case_options=self.case_options)

    # Only generate forecasts and targets if updating cases are investigated
    if self.case_options.get('update_mode') != 'NO_UPDATE':
      # Object used to generate forecasts
      Forecasts = ConstructForecast(output_dir=self.output_dir, case_options=self.case_options)

      # Revenue Target
      Target = RevenueTarget(case_options=self.case_options)

      # Defines if renewables are eligible for payments under scheme
      Eligibility = RenewablesEligibility(case_options=self.case_options)

    if self.case_options.get('shock_option') != 'NO_SHOCKS':
      # Object used to generate emissions intensity shock
      Shock = ApplyShock(output_dir=self.output_dir, case_options=self.case_options)

    # Run scenarios
    # -------------
    # Calibration intervals for which model will be run (can run for less than one year by adjusting model_horizon)
    # Note: weekly calibration intervals are used in this analysis
    calibration_intervals = range(1, self.case_options.get('model_horizon') + 1)

    # For each calibration interval
    for calibration_interval in calibration_intervals:
      # Start clock to see how long it takes to solve all scenarios for each calibration interval
      t0 = time.time()

      # Initialise policy parameters if the first calibration interval
      if calibration_interval == calibration_intervals[0]:
        # Initialise permit price
        permit_price = self.case_options.get('initial_permit_price')

        # Initialise rolling scheme revenue (value at the end of previous calibration interval)
        Results.calibration_interval_metrics['rolling_scheme_revenue_interval_start'][calibration_interval] = self.case_options.get('initial_rolling_scheme_revenue')

      # Update rolling scheme revenue (amount of money in bank account at start of interval)
      else:
        Results.calibration_interval_metrics['rolling_scheme_revenue_interval_start'][calibration_interval] = Results.calibration_interval_metrics['rolling_scheme_revenue_interval_end'][calibration_interval - 1]

      try:
        print(f'\n Interval start target revenue: {Target.revenue_target}')
      except:
        pass

      try:
        print(f"Interval start scheme revenue: {Results.calibration_interval_metrics['rolling_scheme_revenue_interval_start'][calibration_interval]}")
      except:
        pass

      # Compute baseline
      # ----------------
      if self.case_options.get('update_mode') == 'NO_UPDATE':
        baseline = self.case_options.get('default_baseline')

      elif self.case_options.get('update_mode') == 'REVENUE_REBALANCE_UPDATE':
        # Re-balance revenue in-balance in following calibration interval by adjust baseline
        baseline = Baseline.get_revenue_rebalance_update(case_options=self.case_options,
                                                         model_object=DCOPF,
                                                         calibration_interval=calibration_interval,
                                                         forecast_generator_energy=Forecasts.generator_energy,
                                                         forecast_emissions_intensities=Forecasts.generator_emissions_intensities,
                                                         forecast_intermittent_energy=Forecasts.intermittent_energy,
                                                         renewables_eligibility=Eligibility.renewables_eligibility[calibration_interval],
                                                         target_scheme_revenue=Target.revenue_target,
                                                         permit_price=self.case_options.get('initial_permit_price'),
                                                         scheme_revenue_interval_start=Results.calibration_interval_metrics['rolling_scheme_revenue_interval_start'][calibration_interval])

      elif self.case_options.get('update_mode') == 'MPC_UPDATE':
        # Initialise baseline if first calibration interval
        if calibration_interval == 1:
          baseline = self.case_options.get('default_baseline')

        # Use Model Predictive Controller to update baseline
        baseline = Baseline.get_mpc_update(model_object=MPC,
                                           calibration_interval=calibration_interval,
                                           case_options=self.case_options,
                                           forecast_generator_energy=Forecasts.generator_energy,
                                           forecast_intermittent_energy=Forecasts.intermittent_energy,
                                           forecast_emissions_intensities=Forecasts.generator_emissions_intensities,
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

      # Compute calibration interval (week) metrics
      Results.store_calibration_interval_metrics(model_object=DCOPF,
                                                 calibration_interval=calibration_interval,
                                                 case_options=self.case_options)

      print(f"\n Interval end rolling scheme revenue: {Results.calibration_interval_metrics['rolling_scheme_revenue_interval_end'][calibration_interval]}")

      print(f"Inteval end total emissions {calibration_interval}: {Results.calibration_interval_metrics['total_emissions_tCO2'][calibration_interval]}")

      print(f"Interval end ispatchable Generator energy {calibration_interval}: {Results.calibration_interval_metrics['total_dispatchable_generator_energy_MWh'][calibration_interval]}")

      print(f"Interval end total intermittent energy {calibration_interval}: {Results.calibration_interval_metrics['total_intermittent_energy_MWh'][calibration_interval]}")

      print(f'Interval end baseline applied: {baseline}')

      print(f"Interval end regulated generator emissions intensity: {Results.calibration_interval_metrics['average_emissions_intensity_regulated_generators'][calibration_interval]}")

      print(f'Completed calibration interval {calibration_interval} in {time.time()-t0:.2f}s')

    # Save results
    # ------------
    Results.save_results(output_dir=self.output_dir, model_object=DCOPF)

    return Results.case_summary['case_id']
