"""Class to generate forcecasts"""

import os
import pickle

import numpy as np
import pandas as pd


class ConstructForecast:
    def __init__(self, output_dir):
        # Directory containing benchmark case results. Using these data to construct forecasts signals.
        self.output_dir = output_dir

    def _load_week_metrics(self, run_id):
        "Load week metrics for given run_id"

        # Load weekly results for given run_id
        with open(os.path.join(self.output_dir, f'{run_id}_week_metrics.pickle'), 'rb') as f:
            week_metrics = pickle.load(f)

        return week_metrics

    def _get_model_horizon(self, run_id):
        "Number of intervals in model horizon for given run_id"

        # Load run summary for given run_id
        with open(os.path.join(self.output_dir, f'{run_id}_run_summary.pickle'), 'rb') as f:
            run_summary = pickle.load(f)

        # Max number of intervals for which forecast can be constructed (based on benchmark case that
        # produces same energy and emissions profile)
        model_horizon = run_summary[f'{run_id}']['model_horizon']

        return model_horizon

    def _get_run_summaries(self):
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

        return pd.DataFrame(run_summaries).T

    def _get_benchmark_run_id(self, shock_option, initial_permit_price, model_horizon):
        """Identify run ID used for forecasts based on model parameters"""

        # Summary of all models which have been run
        run_summaries = self._get_run_summaries()

        # Filter benchmark cases corresponding to the given situation
        mask = ((run_summaries['shock_option'] == shock_option)
                & (run_summaries['update_mode'] == 'NO_UPDATE')
                & (run_summaries['initial_permit_price'] == initial_permit_price)
                & (run_summaries['model_horizon'] == model_horizon))

        # Check there is only one matching run ID
        if len(run_summaries.loc[mask]) != 1:
            raise(Exception(f'Should only encounter one run_id with given parameters, encountered: {run_summaries.loc[mask]}'))
        else:
            # Run_id for the benchmark case that will be used to generate forecast signals
            forecast_run_id = run_summaries.loc[mask].index[0]

            return forecast_run_id

    def get_forecast(self, forecast_type, forecast_intervals, forecast_uncertainty_increment, shock_option, initial_permit_price, model_horizon):
        """Get perfect forecasts for dispatchable generater energy, emissions intensities, and
        intermittent energy based on benchmark model results

        Parameters
        ----------
        run_id : str
            ID for run already completed that will result in same emissions / energy output profiles

        forecast_intervals : int
            Number of intervals the forecast should be constructed for

        Returns
        -------
        forecast_generator_energy : dict
            Perfect forecast for generator energy output in each week
        """

        # Benchmark case corresponding to model parameters
        benchmark_run_id = self._get_benchmark_run_id(shock_option, initial_permit_price, model_horizon)

        # Total number of intervals in model horizon
        model_horizon = self._get_model_horizon(benchmark_run_id)

        # Week metrics for benchmark case
        benchmark_week_metrics = self._load_week_metrics(run_id=benchmark_run_id)

        if forecast_type == 'generator_energy':
            # Forecast dispatchable generator energy
            forecast = {week:
                        {j + 1:
                         {duid: np.random.uniform(1 - (forecast_uncertainty_increment * (j + 1)), 1 + (forecast_uncertainty_increment * (j + 1))) * output for duid, output in benchmark_week_metrics['dispatchable_generator_energy_MWh'][week + j].items()}
                         for j in range(0, forecast_intervals)}
                        for week in range(1, model_horizon + 1 - forecast_intervals + 1)}

        elif forecast_type == 'generator_emissions_intensities':
            forecast = {i:
                        {j + 1: benchmark_week_metrics['dispatchable_generator_emissions_intensities'][i + j] for j in range(0, forecast_intervals)}
                        for i in range(1, model_horizon + 1 - forecast_intervals + 1)}

        elif forecast_type == 'intermittent_energy':
            forecast = {i:
                        {j + 1: np.random.uniform(1 - (forecast_uncertainty_increment * (j + 1)), 1 + (forecast_uncertainty_increment * (j + 1))) * benchmark_week_metrics['total_intermittent_energy_MWh'][i + j] for j in range(0, forecast_intervals)} for i in range(1, model_horizon + 1 - forecast_intervals + 1)}
        else:
            raise(Exception(f'Unexpected forecast_type encountered: {forecast_type}'))

        return forecast

    def get_incorrect_forecasts(self, shock_option, week_of_shock, initial_permit_price, model_horizon, forecast_intervals, forecast_type, forecast_uncertainty_increment):
        "Unforseen shock occurs"

        # Benchmark case where no shock occurs
        no_shock_run_id = self._get_benchmark_run_id(shock_option='NO_SHOCKS', initial_permit_price=initial_permit_price, model_horizon=model_horizon)

        # Benchmark case where shock occurs
        shock_run_id = self._get_benchmark_run_id(shock_option=shock_option, initial_permit_price=initial_permit_price, model_horizon=model_horizon)

        # Perturbed forecasts for both cases
        # ----------------------------------
        if forecast_type == 'generator_energy':
            # Generator energy - no shock
            no_shock_forecast = self.get_perturbed_generator_energy_forecast(benchmark_run_id=no_shock_run_id, forecast_intervals=forecast_intervals, forecast_uncertainty_increment=forecast_uncertainty_increment)

            # Generator energy - with shock
            shock_forecast = self.get_perturbed_generator_energy_forecast(benchmark_run_id=shock_run_id, forecast_intervals=forecast_intervals, forecast_uncertainty_increment=forecast_uncertainty_increment)

        elif forecast_type == 'generator_emissions_intensities':
            # Generator energy - no shock
            no_shock_forecast = self.get_perfect_generator_emissions_intensities_forecast(benchmark_run_id=no_shock_run_id, forecast_intervals=forecast_intervals)

            # Generator energy - with shock
            shock_forecast = self.get_perfect_generator_emissions_intensities_forecast(benchmark_run_id=shock_run_id, forecast_intervals=forecast_intervals)

        elif forecast_type == 'intermittent_energy':
            # Generator energy - no shock
            no_shock_forecast = self.get_perturbed_intermittent_energy_forecast(benchmark_run_id=no_shock_run_id, forecast_intervals=forecast_intervals, forecast_uncertainty_increment=forecast_uncertainty_increment)

            # Generator energy - with shock
            shock_forecast = self.get_perturbed_intermittent_energy_forecast(benchmark_run_id=shock_run_id, forecast_intervals=forecast_intervals, forecast_uncertainty_increment=forecast_uncertainty_increment)
        else:
            raise(Exception(f'Unexpected forecast_type encountered: {forecast_type}. Must be either "generator_energy", "generator_emissions_intensities", or "intermittent_energy"'))

        # Replace the forseen shock forecast value with unforeseen shock forecast values up until the week of the shock
        for i in range(1, week_of_shock + 1):
            shock_forecast[i] = no_shock_forecast[i]

        return shock_forecast
