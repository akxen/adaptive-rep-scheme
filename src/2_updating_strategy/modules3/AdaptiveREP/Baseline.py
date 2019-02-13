
class Updater:
    "Compute updates to emissions intensity baseline"

    def __init__(self):
        pass

    def get_revenue_rebalance_update(self, case_options, model_object, calibration_interval, forecast_generator_energy, forecast_emissions_intensities, forecast_intermittent_energy, renewables_eligibility, target_scheme_revenue, permit_price, scheme_revenue_interval_start):
        "Compute baseline to implement given case options"

        # Aggregate forecast statistics
        # -----------------------------
        # Forecast total emissions in next calibration [tCO2]
        total_emissions = sum(forecast_generator_energy[calibration_interval][1][g] * forecast_emissions_intensities[calibration_interval][1][g] for g in model_object.model.OMEGA_G)

        print(f'\n Update forecast total emissions for interval {calibration_interval}: {total_emissions}')

        # Forecast regulated generator energy
        regulated_generator_energy = sum(forecast_generator_energy[calibration_interval][1][g] for g in model_object.model.OMEGA_G)

        # Forecast energy from intermittent reneweable generators (possibly under policy's remit)
        intermittent_energy = forecast_intermittent_energy[calibration_interval][1]

        print(f'Update forecast generator intermittent energy for interval {calibration_interval}: {intermittent_energy}')

        # Forecast regulated generator energy in next period. Value may change depending on whether or not
        # intermittent generators are subject to the scheme's remit.

        # Indicates if renewables are eligible for payments (eligible = True, ineligible = False)
        if renewables_eligibility == 'eligible':
            # Intermittent generators are part of scheme's remit, and receive payments
            regulated_generator_energy += intermittent_energy

        print(f'Update forecast regulated generator energy {calibration_interval}: {regulated_generator_energy}')

        # Forecast regulated generator average emissions intensity
        average_emissions_intensity = total_emissions / regulated_generator_energy

        print(f'Update forecast average emissions intensity {calibration_interval}: {average_emissions_intensity}')

        print(f'Update forecast target scheme revenue {calibration_interval}: {target_scheme_revenue[calibration_interval][1]}')

        print(f'Update forecast scheme revenue interval start {calibration_interval}: {scheme_revenue_interval_start}')

        print(f'Update forecast permit price {calibration_interval}: {permit_price}')

        print(f'Update forecast regulated generator energy (in formula) {calibration_interval}: {regulated_generator_energy}')

        # Update baseline seeking to re-balance net scheme revenue every period based on forecast output
        baseline = average_emissions_intensity - ((target_scheme_revenue[calibration_interval][1] - scheme_revenue_interval_start) / (permit_price * regulated_generator_energy))

        # Set baseline to 0 if updated value less than zero
        if baseline < 0:
            baseline = 0

        print(f'Update baseline: {baseline} \n')

        return baseline

    def get_mpc_update(self, model_object, calibration_interval, case_options, forecast_generator_energy, forecast_intermittent_energy, forecast_emissions_intensities, renewables_eligibility, permit_price, baseline_interval_start, scheme_revenue_interval_start, target_scheme_revenue):
        # Update baseline using a Model Predictive Control paradigm. Goal is to minimise control
        # action (movement of baseline) while achieving target scheme revenue 6 weeks in the future.

        # If first week set baseline to default_baseline
        if calibration_interval == 1:
            baseline = case_options.get('default_baseline')

        # Have limited information at end of model horizon to make forecast e.g. in the 2nd to last
        # week it's not possible to forecast 6 weeks in the future. Instead use the baselines
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

            # Baseline to be implemented next
            baseline = baseline_path[1]

        # End of model horizon (forecast now extends beyond model horizon)
        else:
            # Last calibration interval for which a forecast can be made (forecase extends beyond model horizon)
            last_interval = case_options.get('model_horizon') - case_options.get('forecast_intervals') + 1

            # Update MPC controller parameters -
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

            # Find index corresponding to baseline / indicator when at end of model horizon
            path_index = calibration_interval - (case_options.get('model_horizon') - case_options.get('forecast_intervals'))

            # Baseline to be implemented
            baseline = baseline_path[path_index]

        return baseline
