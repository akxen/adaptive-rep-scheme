import time

from .MPC import MPCModel
from .DCOPF import DCOPFModel
from .Forecast import ConstructForecast
from .ModelUtils import RecordResults, check_case_options, apply_emissions_intensity_shock
from .UpdateBaseline import Updater


def run_case(data_dir, scenarios_dir, output_dir, **kwargs):
    """Run model with given parameters

    Parameters
    ----------
    data_dir : str
        Path to directory containing core data files used in model initialisation

    scenarios_dir : str
        Path to directory containing representative operating scenarios

    output_dir : str
        Path to directory where results files will be written

    **kwargs : dict
        Case options

    Returns
    -------
    case_id : str
        ID used to identify model run
    """

    # Print model being run
    print(f'Running case: {kwargs}')

    # Check if model options valid
    check_case_options(kwargs)

    # Create model objects
    # --------------------
    # Instantiate DCOPF model object
    DCOPF = DCOPFModel(data_dir=data_dir, scenarios_dir=scenarios_dir)

    # Instantiate Model Predictive Controller if MPC update specified
    if kwargs.get('update_mode') == 'MPC_UPDATE':
        MPC = MPCModel(generator_index=DCOPF.model.OMEGA_G, forecast_interval=kwargs.get('forecast_intervals_mpc'))

    # Object used to compute baseline updates
    BaselineUpdater = Updater(output_dir=output_dir)

    # Object used to record model results and compute calibration interval metrics
    Results = RecordResults(case_options=kwargs)

    # Run scenarios
    # -------------
    # Calibration intervals for which model will be run (can run for less than one year by adjusting model_horizon)
    # Note: weekly calibration intervals are used in this analysis
    calibration_intervals = DCOPF.df_scenarios.columns.levels[0][:kwargs.get('model_horizon')]

    # For each calibration interval
    for calibration_interval in calibration_intervals:
        # Start clock to see how long it takes to solve all scenarios for each week
        t0 = time.time()

        # Initialise policy parameters if the first week
        if calibration_interval == calibration_intervals[0]:
            # Initialise permit price
            permit_price = kwargs.get('initial_permit_price')

            # Initialise rolling scheme revenue (value at the end of previous calibration interval)
            Results.calibration_interval_metrics['rolling_scheme_revenue_interval_start'][calibration_interval] = kwargs.get('initial_rolling_scheme_revenue')

        # Update rolling scheme revenue (amount of money in bank account at start of week)
        else:
            Results.calibration_interval_metrics['rolling_scheme_revenue_interval_start'][calibration_interval] = Results.calibration_interval_metrics['rolling_scheme_revenue_interval_end'][calibration_interval - 1]

        # Compute baseline
        # ----------------
        if kwargs.get('update_mode') == 'NO_UPDATE':
            baseline = kwargs.get('default_baseline')

        elif kwargs.get('update_mode') == 'REVENUE_REBALANCE_UPDATE':
            # Construct forecasts
            # -------------------
            generator_output = self.get_forecast(forecast_type='generator_energy', forecast_intervals=case_options.get('forecast_intervals'), forecast_uncertainty_increment=case_options.get('forecast_uncertainty_increment'), shock_option=case_options.get('shock_option'), initial_permit_price=case_options.get('initial_permit_price'), model_horizon=case_options.get('model_horizon'))

            generator_emissions_intensities = self.get_forecast(forecast_type='generator_emissions_intensities', forecast_intervals=case_options.get('forecast_intervals'), forecast_uncertainty_increment=case_options.get('forecast_uncertainty_increment'), shock_option=case_options.get('shock_option'), initial_permit_price=case_options.get('initial_permit_price'), model_horizon=case_options.get('model_horizon'))

            intermittent_energy = self.get_forecast(forecast_type='intermittent_energy', forecast_intervals=case_options.get('forecast_intervals'), forecast_uncertainty_increment=case_options.get('forecast_uncertainty_increment'), shock_option=case_options.get('shock_option'), initial_permit_price=case_options.get('initial_permit_price'), model_horizon=case_options.get('model_horizon'))

            baseline = BaselineUpdater.get_revenue_rebalance_update(case_options=kwargs)

        elif kwargs.get('update_mode') == 'MPC_UPDATE':
            baseline = get_mpc_update(case_options=kwargs, model_object=MPC)

        # Record baseline applying for the given calibration interval
        Results.calibration_interval_metrics['baseline'][calibration_interval] = baseline

        # Apply emissions intensity shock for given calibration interval if shock option specified
        if (kwargs.get('shock_option') == 'EMISSIONS_INTENSITY_SHOCK') and (calibration_interval == kwargs.get('calibration_interval_shock_index')):
            # Update emissions intensities in DCOPF model object
            DCOPF = apply_emissions_intensity_shock(output_dir=output_dir, model_object=DCOPF, case_summary=Results.case_summary)

        # For each representative scenario approximating a calibration interval's operating state
        for scenario_index in DCOPF.df_scenarios.columns.levels[1]:
            # Update model parameters
            DCOPF.update_model_parameters(calibration_interval=calibration_interval, scenario_index=scenario_index, baseline=baseline, permit_price=permit_price)

            # Solve model
            DCOPF.solve_model()

            # Store scenario results
            Results.store_scenario_results(model_object=DCOPF, calibration_interval=calibration_interval, scenario_index=scenario_index)

        # Compute calibration interval (week) metrics
        Results.store_calibration_interval_metrics(model_object=DCOPF, calibration_interval=calibration_interval, case_options=kwargs)

        print(f'Completed calibration interval {calibration_interval} in {time.time()-t0:.2f}s')

    # Save results
    # ------------
    Results.save_results(output_dir=output_dir, model_object=DCOPF)

    return Results.case_summary['case_id']
