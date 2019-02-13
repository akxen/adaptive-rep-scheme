"""Class to generate forcecasts"""

import os
import pickle

import numpy as np
import pandas as pd


class ConstructForecast:
    def __init__(self, output_dir, case_options):
        # Directory containing benchmark case results. Using these data to construct forecasts signals.
        self.output_dir = output_dir

        # Options specified for case
        self.case_options = case_options

        # Generator energy forecast
        self.generator_energy = self.get_forecast('generator_energy')

        # Generator emissions intensity forecast
        self.generator_emissions_intensities = self.get_forecast('generator_emissions_intensities')

        # Intermittent energy
        self.intermittent_energy = self.get_forecast('intermittent_energy')

    def _load_calibration_interval_metrics(self, case_id):
        "Load week metrics for given case_id"

        # Load weekly results for given case_id
        with open(os.path.join(self.output_dir, f'{case_id}_calibration_interval_metrics.pickle'), 'rb') as f:
            week_metrics = pickle.load(f)

        return week_metrics

    def _get_model_horizon(self, case_id):
        "Number of intervals in model horizon for given case_id"

        # Load run summary for given case_id
        with open(os.path.join(self.output_dir, f'{case_id}_case_summary.pickle'), 'rb') as f:
            case_summary = pickle.load(f)

        # Max number of intervals for which forecast can be constructed (based on benchmark case that
        # produces same energy and emissions profile)
        model_horizon = case_summary[f'{case_id}']['model_horizon']

        return model_horizon

    def _get_case_summaries(self):
        """Collate information summarising the parameters used for each model

        Returns
        -------
        case_summaries : dict
            Dictionary summarising model parameterisations
        """

        # Find all results summary files
        case_summary_files = [i for i in os.listdir(self.output_dir) if 'case_summary' in i]

        # Container for dictionaries summarising model runs
        case_summaries = []

        # Open each run summary file and compile in a single dictionary
        for i in case_summary_files:
            with open(os.path.join(self.output_dir, i), 'rb') as f:
                # Load run summary from file
                case_summary = pickle.load(f)

                # Append to dictionary collating all run summaries
                case_summaries.append(case_summary)

        return pd.DataFrame(case_summaries).set_index('case_id')

    def _get_benchmark_case_id(self, shock_option):
        """Identify run ID used for forecasts based on model parameters"""

        # Summary of all models which have been run
        case_summaries = self._get_case_summaries()

        # Filter benchmark cases corresponding to the given situation
        mask = ((case_summaries['shock_option'] == shock_option)
                & (case_summaries['update_mode'] == 'NO_UPDATE')
                & (case_summaries['initial_permit_price'] == self.case_options.get('initial_permit_price'))
                & (case_summaries['model_horizon'] == self.case_options.get('model_horizon')))

        # Check there is only one matching run ID
        if len(case_summaries.loc[mask]) != 1:
            raise(Exception(f'Should only encounter one case_id with given parameters, encountered: {case_summaries.loc[mask]}'))
        else:
            # case_id for the benchmark case that will be used to generate forecast signals
            forecast_case_id = case_summaries.loc[mask].index[0]
            print(f'Forecast benchmark case ID: {forecast_case_id}')

            return forecast_case_id

    def _get_benchmark_forecast(self, forecast_type, benchmark_case_id):
        """Get forecasts for dispatchable generater energy, emissions intensities, and
        intermittent energy based on benchmark model results

        Parameters
        ----------
        case_id : str
            ID for run already completed that will result in same emissions / energy output profiles

        forecast_intervals : int
            Number of intervals the forecast should be constructed for

        Returns
        -------
        forecast_generator_energy : dict
            Perfect forecast for generator energy output in each week
        """

        # Week metrics for benchmark case
        benchmark_calibration_interval_metrics = self._load_calibration_interval_metrics(case_id=benchmark_case_id)

        if forecast_type == 'generator_energy':
            # Initialise dictionary used to contain forecast values
            forecast = dict()

            # For each calibration interval
            for calibration_interval in range(1, self.case_options.get('model_horizon') + 1 - self.case_options.get('forecast_intervals') + 1):
                forecast[calibration_interval] = dict()

                # For each forecast interval (lookin)
                for forecast_interval in range(1, self.case_options.get('forecast_intervals') + 1):
                    forecast[calibration_interval][forecast_interval] = dict()

                    # For each generator, construct a forecast value for energy output by perturbing
                    # output observed in the benchmark case
                    for generator, output in benchmark_calibration_interval_metrics['dispatchable_generator_energy_MWh'][calibration_interval + forecast_interval - 1].items():
                        forecast[calibration_interval][forecast_interval][generator] = np.random.uniform(1 - (self.case_options.get('forecast_uncertainty_increment') * forecast_interval), 1 + (self.case_options.get('forecast_uncertainty_increment') * forecast_interval)) * output

        elif forecast_type == 'generator_emissions_intensities':
            # Initialise dictionary used to contain forecast values
            forecast = dict()

            # For each calibration interval
            for calibration_interval in range(1, self.case_options.get('model_horizon') + 1 - self.case_options.get('forecast_intervals') + 1):
                forecast[calibration_interval] = dict()

                # For each forecast interval
                for forecast_interval in range(1, self.case_options.get('forecast_intervals') + 1):
                    forecast[calibration_interval][forecast_interval] = benchmark_calibration_interval_metrics['dispatchable_generator_emissions_intensities'][calibration_interval + forecast_interval - 1]

        elif forecast_type == 'intermittent_energy':
            # Initialise dictionary used to contain forecast values
            forecast = dict()

            # For each calibration interval
            for calibration_interval in range(1, self.case_options.get('model_horizon') + 1 - self.case_options.get('forecast_intervals') + 1):
                forecast[calibration_interval] = dict()

                # For each forecast interval
                for forecast_interval in range(1, self.case_options.get('forecast_intervals') + 1):
                    forecast[calibration_interval][forecast_interval] = benchmark_calibration_interval_metrics['total_intermittent_energy_MWh'][calibration_interval + forecast_interval - 1] * np.random.uniform(1 - (self.case_options.get('forecast_uncertainty_increment') * forecast_interval), 1 + (self.case_options.get('forecast_uncertainty_increment') * forecast_interval))

        else:
            raise(Exception(f'Unexpected forecast_type encountered: {forecast_type}'))

        return forecast

    def _overlap_forecasts(self, forecast_1, forecast_2, overlap_intervals):
        "Overlap values from forecast 1 onto forecast 2"

        # Assign values from forecast 1 to values for forecast 2 for overlap_intervals
        for i in range(1, overlap_intervals + 1):
            forecast_2[i] = forecast_1[i]

        return forecast_2

    def get_forecast(self, forecast_type):
        "Get forecast"

        # Benchmark case corresponding to model parameters
        benchmark_case_id = self._get_benchmark_case_id(shock_option=self.case_options.get('shock_option'))

        # Forecast based on benchmark corresponding to case options
        forecast = self._get_benchmark_forecast(forecast_type, benchmark_case_id)

        # If a forecast shock is specified
        if ('forecast_shock' in self.case_options) and (self.case_options.get('forecast_shock')):
            # Benchmark case corresponding to no-shock benchmark case
            benchmark_case_id_no_shock = self._get_benchmark_case_id(shock_option='NO_SHOCKS')

            # No shock forecast
            forecast_no_shock = self._get_benchmark_forecast(forecast_type, benchmark_case_id_no_shock)

            # Overlap forecasts. Shock is unforseen.
            forecast = self._overlap_forecasts(forecast_no_shock, forecast, self.case_options.get('shock_index'))

        return forecast
