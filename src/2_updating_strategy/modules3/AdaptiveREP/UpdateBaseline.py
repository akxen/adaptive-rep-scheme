
class Updater:
    "Compute updates to emissions intensity baseline"

    def __init__(self, output_dir):
        super().__init__(output_dir)

    def get_revenue_rebalance_update(self, case_options, model_object, calibration_interval, forecast_generator_energy, forecast_generator_emissions_intensities, forecast_intermittent_energy, target_scheme_revenue, permit_price, scheme_revenue_interval_start):
        "Compute baseline to implement given case options"

        # Aggregate forecast statistics
        # -----------------------------
        # Forecast total emissions in next calibration [tCO2]
        total_emissions = sum(forecast_generator_energy[calibration_interval][1][g] * forecast_generator_emissions_intensities[calibration_interval][1][g] for g in model_object.model.OMEGA_G)

        # Forecast regulated generator energy
        regulated_generator_energy = sum(forecast_generator_energy[calibration_interval][1][g] for g in model_object.model.OMEGA_G)

        # Forecast energy from intermittent reneweable generators (possibly under policy's remit)
        intermittent_energy = forecast_intermittent_energy[calibration_interval][1]

        # Forecast regulated generator energy in next period. Value may change depending on whether or not
        # intermittent generators are subject to the scheme's remit.

        # Indicates if renewables are eligible for payments (default=False)
        if 'intermittent_generators_eligible' in case_options:
            renewables_eligible_this_week = case_options.get('intermittent_generators_eligible')[calibration_interval][1]
        else:
            renewables_eligible_this_week = False

        if renewables_eligible_this_week:
            # Intermittent generators are part of scheme's remit, and receive payments
            regulated_generator_energy += intermittent_energy

        # Forecast regulated generator average emissions intensity
        average_emissions_intensity = regulated_generator_energy / total_emissions

        # Update baseline seeking to re-balance net scheme revenue every period based on forecast output
        baseline = average_emissions_intensity - ((target_scheme_revenue[calibration_interval][1] - scheme_revenue_interval_start) / (permit_price * regulated_generator_energy))

        # Set baseline to 0 if updated value less than zero
        if baseline < 0:
            baseline = 0

        return baseline

    def get_mpc_update(case_options):
        # Update baseline using a Model Predictive Control paradigm. Goal is to minimise control
        # action (movement of baseline) while achieving target scheme revenue 6 weeks in the future.

        # If first week set baseline to default_baseline
        if week_index == 1:
            baseline = kwargs.get('default_baseline')

        # Have limited information at end of model horizon to make forecast e.g. in the 2nd to last
        # week it's not possible to forecast 6 weeks in the future. Instead use the baselines
        # generated from the last optimal path calculated.
        if week_index <= weeks[-1] - kwargs.get('forecast_interval_mpc') + 1:
            # Expected generator energy output in forecast interval
            forecast_energy = kwargs.get('forecast_generator_energy')[week_index]

            # Expected generator emissions intensities in forecast interval
            forecast_emissions_intensity = kwargs.get('forecast_generator_emissions_intensity')[week_index]

            # Expected energy output from renewables over forecast interval
            forecast_intermittent_energy = kwargs.get('forecast_intermittent_generator_energy')[week_index]

            # Intermittent generators included this week
            renewables_included_this_week = kwargs.get('intermittent_generators_regulated')[week_index][1]

            # Update MPC controller parameters
            MPC.update_model_parameters(forecast_generator_emissions_intensity=forecast_emissions_intensity,
                                        forecast_generator_energy=forecast_energy,
                                        forecast_intermittent_energy=forecast_intermittent_energy,
                                        intermittent_generators_regulated=kwargs.get('intermittent_generators_regulated')[week_index],
                                        permit_price=permit_price,
                                        initial_emissions_intensity_baseline=baseline,
                                        initial_rolling_scheme_revenue=rolling_scheme_revenue_interval_start,
                                        target_rolling_scheme_revenue=kwargs.get('target_scheme_revenue')[week_index][kwargs.get('forecast_interval_mpc')])

            # Solve model
            MPC.solve_model()

            # Get next baseline to be implemented
            baseline_path = MPC.get_optimal_baseline_path()

            # Baseline to be implemented next
            baseline = baseline_path[1]

        else:
            # Find index of corresponding to baseline / indicator when at end of model horizon
            path_index = week_index - (weeks[-1] - kwargs.get('forecast_interval_mpc'))

            # Baseline to be implemented
            baseline = baseline_path[path_index]

            # Indicates if renewables receive payments this week
            renewables_included_this_week = kwargs.get('intermittent_generators_regulated')[weeks[-1] - kwargs.get('forecast_interval_mpc') + 1][path_index]
