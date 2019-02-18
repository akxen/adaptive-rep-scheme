
class Updater:
    "Compute updates to emissions intensity baseline"

    def get_revenue_rebalance_update(self, model_object, case_options, calibration_interval, forecast_generator_energy, forecast_emissions_intensities, forecast_intermittent_energy, renewables_eligibility, permit_price, scheme_revenue_interval_start, target_scheme_revenue):
        """Compute next baseline in sequence based on forecast values, current state of system, and case options

        Parameters
        ----------
        model_object : pyomo model object
            Power flow model for given calibration interval

        case_options : dict
            Options defining case parameters

        calibration_interval : int
            Index of calibration interval for which the update should be generated

        forecast_generator_energy : dict
            Forecast energy output from generators covered by emissions policy

        forecast_emissions_intensities : dict
            Forecast emissions intensities for generators covered by emissions policy

        forecast_intermittent_energy : dict
            Forecast energy output from intermittent renewables

        renewables_eligibility : bool
            Indicates if renewables are eligible for emissions payements in next calibration interval

        permit_price : float
            Permit price applied for next calibration interval

        scheme_revenue_interval_start : float
            Scheme revenue at start of calibration interval

        target_scheme_revenue : dict
            Target scheme revenue in future calibration intervals


        Returns
        -------
        baseline : float
            Value emissions intensity baseline should be set to in following calibration interval
        """

        # Aggregate forecast statistics
        # -----------------------------
        # Forecast total emissions in next calibration [tCO2]
        total_emissions = sum(forecast_generator_energy[calibration_interval][1][g] * forecast_emissions_intensities[calibration_interval][1][g] for g in model_object.model.OMEGA_G)

        # Forecast energy output from generators covered by emissions policy [MWh]
        regulated_generator_energy = sum(forecast_generator_energy[calibration_interval][1][g] for g in model_object.model.OMEGA_G)

        # Forecast energy from intermittent reneweable generators (possibly under policy's remit)
        intermittent_energy = forecast_intermittent_energy[calibration_interval][1]

        # Indicates if renewables are eligible for payments (eligible = True, ineligible = False)
        if renewables_eligibility[1]:
            # Intermittent generators are under scheme's remit, and receive payments
            regulated_generator_energy += intermittent_energy

        # Forecast regulated generator average emissions intensity
        average_emissions_intensity = total_emissions / regulated_generator_energy

        # Update baseline seeking to re-balance net scheme revenue in the next period based on forecast values
        baseline = average_emissions_intensity - ((target_scheme_revenue[calibration_interval][1] - scheme_revenue_interval_start) / (permit_price * regulated_generator_energy))

        # Set baseline to 0 if updated value less than zero
        if baseline < 0:
            baseline = 0

        return baseline

    def get_mpc_update(self, model_object, case_options, calibration_interval, forecast_generator_energy, forecast_emissions_intensities, forecast_intermittent_energy, renewables_eligibility, permit_price, baseline_interval_start, scheme_revenue_interval_start, target_scheme_revenue):
        """Update baseline using a Model Predictive Control paradigm. Goal is to minimise control
        action (movement of baseline) while achieving target scheme revenue x calibration intervals in the future.

        Parameters
        ----------
        model_object : pyomo model object
            MPC model

        case_options : dict
            Description of case parameters

        calibration_interval : int
            Calibration interval index

        forecast_generator_energy : dict
            Forecast energy output from generators covered by emissions policy

        forecast_emissions_intensities : dict
            Forecast emissions intensities for generators covered by emissions policy

        forecast_intermittent_energy : dict
            Forecast energy output from intermittent renewables

        renewables_eligibility : bool
            Indicates if renewables are eligible for emissions payements in following calibration intervals

        permit_price : float
            Permit price applied for following calibration intervals

        baseline_interval_start : float
            Value baseline was set to in preceding calibration interval

        scheme_revenue_interval_start : float
            Amount of scheme revenue at start of calibration interval

        target_scheme_revenue : dict
            Target scheme revenue in future calibration intervals


        Returns
        -------
        baseline : float
            Value emissions intensity baseline should be set to in following calibration interval
        """

        # If first calibration interval set baseline to default_baseline
        if calibration_interval == 1:
            baseline = case_options.get('default_baseline')

        # Have limited information at end of model horizon to make forecast e.g. in the 2nd to last
        # week it's not possible to forecast 6 weeks in the future. Instead, use the baselines
        # generated from the last optimal path calculated.

        elif calibration_interval <= case_options.get('model_horizon') - case_options.get('forecast_intervals') + 1:
            # Update MPC controller parameters
            model_object.update_model_parameters(forecast_emissions_intensities=forecast_emissions_intensities[calibration_interval],
                                                 forecast_generator_energy=forecast_generator_energy[calibration_interval],
                                                 forecast_intermittent_energy=forecast_intermittent_energy[calibration_interval],
                                                 renewables_eligibility=renewables_eligibility[calibration_interval],
                                                 permit_price=permit_price,
                                                 baseline_interval_start=baseline_interval_start,
                                                 scheme_revenue_interval_start=scheme_revenue_interval_start,
                                                 target_scheme_revenue=target_scheme_revenue[calibration_interval][case_options.get('forecast_intervals')])

            # Solve model
            model_object.solve_model()

            # Path of baselines that achieve target
            baseline_path = model_object.get_optimal_baseline_path()

            # Baseline to be implemented next calibration interval
            baseline = baseline_path[1]

        # End of model horizon (forecast now extends beyond model horizon)
        else:
            # Last calibration interval for which a forecast can be made
            last_interval = case_options.get('model_horizon') - case_options.get('forecast_intervals') + 1

            # Update MPC controller parameters
            model_object.update_model_parameters(forecast_emissions_intensities=forecast_emissions_intensities[last_interval],
                                                 forecast_generator_energy=forecast_generator_energy[last_interval],
                                                 forecast_intermittent_energy=forecast_intermittent_energy[last_interval],
                                                 renewables_eligibility=renewables_eligibility[last_interval],
                                                 permit_price=permit_price,
                                                 baseline_interval_start=baseline_interval_start,
                                                 scheme_revenue_interval_start=scheme_revenue_interval_start,
                                                 target_scheme_revenue=target_scheme_revenue[last_interval][case_options.get('forecast_intervals')])

            # Path of baselines that achieve target
            baseline_path = model_object.get_optimal_baseline_path()

            # Find index corresponding to baseline in baseline_path that should be used when at end of model horizon
            path_index = calibration_interval - (case_options.get('model_horizon') - case_options.get('forecast_intervals'))

            # Baseline to be implemented
            baseline = baseline_path[path_index]

        return baseline
