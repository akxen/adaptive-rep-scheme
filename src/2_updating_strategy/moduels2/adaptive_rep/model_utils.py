"""Functions to generate forcecasts and format input for model"""

import os
import pickle

import numpy as np


class Forecast:
    def __init__(self, output_dir, shock_option, initial_permit_price, model_horizon, forecast_intervals, forecast_uncertainty_increment):
        # Directory containing benchmark case results. Using these data to construct forecasts signals.
        self.output_dir = output_dir

        self.forecast_intervals = forecast_intervals
        self.forecast_uncertainty_increment = forecast_uncertainty_increment

    def get_run_summaries(self):
        """Collate information summarising the parameters used for each model

        Returns
        -------
        run_summaries : dict
            Dictionary summarising model parameterisations
        """

        # Find all results summary files
        run_summary_files = [i for i in os.listdir(self.output_dir) if 'run_summary' in i]

        # Container for dictionaries summarising model runs
        run_summaries = dict()

        # Open each run summary file and compile in a single dictionary
        for i in run_summary_files:
            with open(os.path.join(self.output_dir, i), 'rb') as f:
                # Load run summary from file
                run_summary = pickle.load(f)

                # Append to dictionary collating all run summaries
                run_summaries = {**run_summaries, **run_summary}

        return run_summaries

    def get_benchmark_run_id(self, no_shocks=False):
        """Identify run ID used for forecasts based on model parameters"""

        # Summary of all models which have been run
        run_summaries = self.get_run_summaries()

        # Check if the no-shocks benchmark case is specified
        if no_shocks:
            shock = 'NO_SHOCKS'
        else:
            shock = self.shock_option

        # Filter benchmark cases corresponding to the given situation
        mask = ((run_summaries['shock_option'] == shock)
                & (run_summaries['update_mode'] == 'NO_UPDATE')
                & (run_summaries['initial_permit_price'] == self.initial_permit_price)
                & (run_summaries['model_horizon'] == self.model_horizon))

        # Check there is only one matching run ID
        if len(run_summaries.loc[mask]) != 1:
            raise(Exception('Should only encounter one run_id with given parameters'))
        else:
            # Get the run_id for the scenario that will be used to generate forecast signals
            forecast_run_id = run_summaries.loc[mask].index[0]

            return forecast_run_id

    def get_perfect_forecasts(self, run_id):
        """Get perfect forecast for energy and emissions intensities based on benchmark model results

        Parameters
        ----------
        output_dir : str
            Directory containing benchmark simulation results

        run_id : str
            ID for run already completed that will result in same emissions / energy output profiles

        forecast_intervals : int
            Number of intervals the forecast should be constructed for

        Returns
        -------
        forecast_generator_energy : dict
            Perfect forecast for generator energy output in each week

        forecast_generator_emissions_intensity : dict
            Perfect forecast for generator emissions intensities in each week
        """

        # Load run summary for given run_id
        with open(os.path.join(self.output_dir, f'{run_id}_run_summary.pickle'), 'rb') as f:
            run_summary = pickle.load(f)

        # Max number of intervals for which forecast can be constructed (based on benchmark case that
        # produces same energy and emissions profile)
        total_intervals = run_summary[f'{run_id}']['model_horizon']

        # Load weekly results for given run_id
        with open(os.path.join(self.output_dir, f'{run_id}_week_metrics.pickle'), 'rb') as f:
            week_metrics = pickle.load(f)

        # Forecast dispatchable generator energy
        forecast_generator_energy = {i: {j + 1: week_metrics['dispatchable_generator_energy_MWh'][i + j] for j in range(0, self.forecast_intervals)} for i in range(1, total_intervals + 1 - self.forecast_intervals + 1)}

        # Forecast generator emissions intensities
        forecast_generator_emissions_intensity = {i: {j + 1: week_metrics['dispatchable_generator_emissions_intensities'][i + j] for j in range(0, self.forecast_intervals)} for i in range(1, total_intervals + 1 - self.forecast_intervals + 1)}

        # Forecast energy output from intermittent sources
        forecast_intermittent_generator_energy = {i: {j + 1: week_metrics['total_intermittent_energy_MWh'][i + j] for j in range(0, self.forecast_intervals)} for i in range(1, total_intervals + 1 - self.forecast_intervals + 1)}

        return forecast_generator_energy, forecast_generator_emissions_intensity, forecast_intermittent_generator_energy

    def get_perturbed_forecasts(self, run_id):
        """Add uncerainty to forecasted values

        Parameters
        ----------
        forecast_type : str
            Type of forecast to be perturbed (underlying forecast has different dictionary structure).
            Options - INTERMITTENT_ENERGY or GENERATOR_ENERGY

        forecast : dict
            Perfect forecast

        forecast_uncertainty_increment : float
            Percentage uncertainty to be used in scaling factor for each week. E.g. if 0.05, then
            the first week's (perfect) forecast will be scaled by a uniformly distributed random number
            in the interval (0.95, 1.05), if the second week it will be scaled by a number in the interval
            (0.9, 1.1) and so on.


        Returns
        -------
        perturbed_forecast : dict
            Forecasted values with some uncertainty
        """

        # Perfect forecasts
        forecast_generator_energy, forecast_generator_emissions_intensity, forecast_intermittent_generator_energy = self.get_perfect_forecasts(run_id=run_id)

        # Generator energy forecast (perturbed)
        forecast_generator_energy_perturbed = {key_1:
                                               {key_2:
                                                {key_3: value_3 * np.random.uniform(1 - (self.forecast_uncertainty_increment * key_2), 1 + (self.forecast_uncertainty_increment * key_2))
                                                 for key_3, value_3 in value_2.items()} for key_2, value_2 in value_1.items()}
                                               for key_1, value_1 in forecast_generator_energy.items()}

        # Intermittent energy forecast (perturbed)
        forecast_intermittent_generator_energy_perturbed = {key_1:
                                                            {key_2: value_2 * np.random.uniform(1 - (forecast_uncertainty_increment * key_1), 1 + (forecast_uncertainty_increment * key_1)) for key_2, value_2 in value_1.items()}
                                                            for key_1, value_1 in forecast.items()}

        return forecast_generator_energy_perturbed, forecast_generator_emissions_intensity, forecast_intermittent_generator_energy_perturbed

    def get_shocked_forecast(self, week_of_shock):
        "Unanticipated shock to forecast"

        # Summary of all model parameterisations
        run_summaries = self.get_run_summaries()

        # Run ID corresponding to benchmark case where there are no shocks
        forecast_run_id_no_shocks = self.get_benchmark_run_id(no_shocks=True)

        # Run ID corresponding to benchmark case (where there is an anticipated shock)
        forecast_run_id_benchmark = self.get_benchmark_run_id()

        # Forecasts when the shock is unforeseen
        forecast_generator_energy_no_shock, forecast_generator_emissions_intensity_no_shock, forecast_intermittent_generator_energy_no_shock = get_perturbed_forecasts(run_id=forecast_run_id_no_shocks)

        # Forecast when the shock is forseen
        forecast_generator_energy, forecast_generator_emissions_intensity, forecast_intermittent_generator_energy = get_perturbed_forecasts(run_id=forecast_run_id_benchmark)

        # Combine forecasts with and without shocks. Use no-shock perturbed forecasts for weeks <= week_of_shock
        # and correct (anticipated shock) forecasts for intervals > week_of_shock
        for i in range(1, self.week_of_shock + 1):
            # Updated generator energy forecast
            forecast_generator_energy[i] = forecast_generator_energy_no_shock_perturbed[i]

            # Updated intermittent generator energy forecast
            forecast_intermittent_generator_energy[i] = forecast_intermittent_generator_energy_no_shock_perturbed[i]

            # Update emissions intensity forecast (policy authority gets it 'wrong' for the week of the shock)
            forecast_generator_emissions_intensity[i] = forecast_generator_emissions_intensity_no_shock[i]

        return forecast_generator_energy, forecast_generator_emissions_intensity, forecast_intermittent_generator_energy


def get_formatted_weekly_input(self, weekly_input, forecast_intervals=1):
    """Format weekly targets / indicator dictionaries so they can be used in revenue rebalancing or MPC updates

    Parameters
    ----------
    weekly_input : dict
        Target or indicator given corresponds to a given week

    forecast_intervals : int
        Number of forecast intervals if using MPC. Default = 1.

    Returns
    -------
    intermittent_generators_regulated : dict
        Formatted dictionary denoting whether renewables are eligible in following periods.
    """

    # Length of model horizon
    model_horizon = len(weekly_input)

    # Formatted so can be used in either revenue re-balancing or MPC update
    formatted_weekly_input = {i: {j + 1: weekly_input[j + i] for j in range(0, self.forecast_intervals)} for i in range(1, self.model_horizon + 1 - self.forecast_intervals + 1)}

    return formatted_weekly_input
