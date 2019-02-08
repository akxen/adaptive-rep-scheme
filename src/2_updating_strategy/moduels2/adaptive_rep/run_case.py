

def run_case(data_dir,
             scenarios_dir,
             run_summaries,
             shock_option,
             update_mode,
             initial_permit_price,
             model_horizon,
             forecast_intervals,
             forecast_uncertainty_increment,
             weekly_target_scheme_revenue,
             renewables_eligibility,
             description,
             week_of_shock,
             forecast_shock,
             default_baseline,
             initial_rolling_scheme_revenue,
             seed):
    """Run different cases / scenarios

    Parameters
    ----------
    data_dir : str
        Directory containing core data files used to construct DCOPF model

    scenarios_dir : str
        Directory containg representative operating scenarios for each week of 2017

    run_summaries : pandas DataFrame
        Summary of benchmark model parameters

    update_mode : str
        Type of baseline update to be implemented
        Options -   NO_UPDATE - no update made to baseline between successive weeks
                    REVENUE_REBALANCE_UPDATE -  baseline updated such that revenue imbalance is corrected in
                                                following week
                    MPC_UPDATE -    Model Predictive Control attempts to adjust baseline such that revenue
                                    imbalance is corrected at end of its forecast horizon. Seeks to mimise
                                    conrol action (changes to baseline) required to do so.

    initial_permit_price : float
        Permit price applying to emissions [$/tCO2]

    model_horizon : int
        Number of time intervals (weeks) the model is run for

    forecast_intervals : int or None
        Number of intervals considered when implementing baseline update [weeks].

    forecast_uncertainty_increment : float or None
        Scaling factor used to perturb perfect forecasts. This scaling factor is multiplied by the number
        of weeks into the future for which the forecast is generated.

    weekly_target_scheme_revenue : dict or None
        Scheme revenue target defined for each week in model horizon

    renewables_eligibility : dict or None
        Defines whether intermittent renewables can receive payments under scheme each week

    description : str
        Description of case being investigated

    week_of_shock : int or None
        Week at which emissions intensity shock is implemented (if specified)

    forecast_shock : bool or None
        Indicates if forecast anticipates the shock in advance (True = shock anticipated,
        False=Shock unanticipated)

    default_baseline : float
        Default value for emissions intensity baseline [tCO2/MWh]

    initial_rolling_scheme_revenue : float
        Initial rolling scheme revenue at scheme start [$]

    seed : int
        Number used to initialise random number generator


    Returns
    -------
    run_id : str
        ID of case being investigated
    """

    # Identify run ID used for forecasts based on model parameters
    mask = ((run_summaries['shock_option'] == shock_option)
            & (run_summaries['update_mode'] == 'NO_UPDATE')
            & (run_summaries['initial_permit_price'] == initial_permit_price)
            & (run_summaries['model_horizon'] == model_horizon))

    # Check there is only one matching run ID
    if len(run_summaries.loc[mask]) != 1:
        raise(Exception('Should only encounter one run_id with given parameters'))
    else:
        # Get the run_id for the scenario that will be used to generate forecast signals
        forecast_run_id = run_summaries.loc[mask].index[0]

    # Perfect forecasts
    forecast_generator_energy, forecast_generator_emissions_intensity, forecast_intermittent_generator_energy = get_perfect_forecasts(run_id=forecast_run_id, forecast_intervals=forecast_intervals)

    # Perturb perfect forecasts by scaling factor
    # -------------------------------------------
    # Note: When uncertainty increment = 0, the perturbed forecast = perfect forecast

    # Perturbed generator energy forecast
    forecast_generator_energy_perturbed = get_perturbed_forecast(forecast_type='GENERATOR_ENERGY', forecast=forecast_generator_energy, forecast_uncertainty_increment=forecast_uncertainty_increment)

    # Perturbed intermittent energy forecast
    forecast_intermittent_generator_energy_perturbed = get_perturbed_forecast(forecast_type='INTERMITTENT_ENERGY', forecast=forecast_intermittent_generator_energy, forecast_uncertainty_increment=forecast_uncertainty_increment)

    # If shock is NOT anticipated in the forecast
    # -------------------------------------------
    if forecast_shock:
        # Identify run ID for the no-shock case
        mask_no_shocks = ((run_summaries['shock_option'] == 'NO_SHOCKS')
                          & (run_summaries['update_mode'] == 'NO_UPDATE')
                          & (run_summaries['initial_permit_price'] == initial_permit_price)
                          & (run_summaries['model_horizon'] == model_horizon))

        # Check there is only one matching run ID
        if len(run_summaries.loc[mask_no_shocks]) != 1:
            raise(Exception('Should only encounter one run_id with given parameters'))
        else:
            # Get the run_id for the scenario that will be used to no-shock forecast signals
            forecast_run_id_no_shocks = run_summaries.loc[mask_no_shocks].index[0]

        # No-shock perfect forecasts
        forecast_generator_energy_no_shock, forecast_generator_emissions_intensity_no_shock, forecast_intermittent_generator_energy_no_shock = get_perfect_forecasts(run_id=forecast_run_id_no_shocks, forecast_intervals=forecast_intervals)

        # No-shock perturbed generator energy forecast
        forecast_generator_energy_no_shock_perturbed = get_perturbed_forecast(forecast_type='GENERATOR_ENERGY', forecast=forecast_generator_energy_no_shock, forecast_uncertainty_increment=forecast_uncertainty_increment)

        # No-shock perturbed intermittent energy forecast
        forecast_intermittent_generator_energy_no_shock_perturbed = get_perturbed_forecast(forecast_type='INTERMITTENT_ENERGY', forecast=forecast_intermittent_generator_energy_no_shock, forecast_uncertainty_increment=forecast_uncertainty_increment)

        # Combine forecasts with and without shocks. Use no-shock perturbed forecasts for weeks <= week_of_shock
        # and correct (anticipated shock) forecasts for intervals > week_of_shock

        for i in range(1, week_of_shock + 1):
            # Updated generator energy forecast
            forecast_generator_energy_perturbed[i] = forecast_generator_energy_no_shock_perturbed[i]

            # Updated intermittent generator energy forecast
            forecast_intermittent_generator_energy_perturbed[i] = forecast_intermittent_generator_energy_no_shock_perturbed[i]

            # Update emissions intensity forecast (policy authority gets it 'wrong' for the week of the shock)
            forecast_generator_emissions_intensity[i] = forecast_generator_emissions_intensity_no_shock[i]

    # Define revenue targets and renewables eligibility
    # -------------------------------------------------
    # Format scheme revenue revenue dictionary so it can be used in model
    target_scheme_revenue = get_formatted_weekly_input(weekly_target_scheme_revenue, forecast_intervals=forecast_intervals)

    # Formatted dictionary indicating renewable generator eligibility
    intermittent_generators_regulated = get_formatted_weekly_input(renewables_eligibility, forecast_intervals=forecast_intervals)

    # Run case
    # --------
    run_id = run_scenarios(data_dir=data_dir,
                           scenarios_dir=scenarios_dir,
                           shock_option=shock_option,
                           update_mode=update_mode,
                           description=description,
                           intermittent_generators_regulated=intermittent_generators_regulated,
                           forecast_generator_emissions_intensity=forecast_generator_emissions_intensity,
                           forecast_generator_energy=forecast_generator_energy_perturbed,
                           forecast_intermittent_generator_energy=forecast_intermittent_generator_energy_perturbed,
                           forecast_uncertainty_increment=forecast_uncertainty_increment,
                           forecast_interval_mpc=forecast_intervals,
                           forecast_shock=forecast_shock,
                           week_of_shock=week_of_shock,
                           default_baseline=default_baseline,
                           initial_permit_price=initial_permit_price,
                           initial_rolling_scheme_revenue=initial_rolling_scheme_revenue,
                           target_scheme_revenue=target_scheme_revenue,
                           seed=seed,
                           model_horizon=model_horizon)

    return run_id
