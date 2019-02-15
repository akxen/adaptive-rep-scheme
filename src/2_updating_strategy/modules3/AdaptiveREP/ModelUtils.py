"""Utilities used when running agent-based simulation"""

import pickle
import hashlib

import numpy as np
import pandas as pd


class RecordResults:
    """Record model results"""

    def __init__(self, case_options):
        """Initialise results containers

        Parameters
        ----------
        case_options : dict
            Dictionary of options specifying case parameters
        """

        # Result containers
        # -----------------
        # Prices at each node for each scenario
        self.scenario_nodal_prices = dict()

        # Power output for each scenario
        self.scenario_power_output = dict()

        # Results for each scenario
        self.scenario_metrics = {'net_scheme_revenue_dispatchable_generators': dict(),
                                 'net_scheme_revenue_intermittent_generators': dict(),
                                 'total_emissions_tCO2': dict(),
                                 'dispatchable_generator_energy_MWh': dict(),
                                 'total_dispatchable_generator_energy_MWh': dict(),
                                 'total_intermittent_energy_MWh': dict(),
                                 'total_demand_MWh': dict(),
                                 'energy_revenue': dict(),
                                 }

        # Aggregated results for each calibration interval
        self.calibration_interval_metrics = {'baseline': dict(),
                                             'net_scheme_revenue_dispatchable_generators': dict(),
                                             'net_scheme_revenue_intermittent_generators': dict(),
                                             'net_scheme_revenue': dict(),
                                             'rolling_scheme_revenue_interval_start': dict(),
                                             'rolling_scheme_revenue_interval_end': dict(),
                                             'dispatchable_generator_emissions_intensities': dict(),
                                             'total_emissions_tCO2': dict(),
                                             'dispatchable_generator_energy_MWh': dict(),
                                             'total_dispatchable_generator_energy_MWh': dict(),
                                             'total_intermittent_energy_MWh': dict(),
                                             'total_demand_MWh': dict(),
                                             'average_emissions_intensity_regulated_generators': dict(),
                                             'average_emissions_intensity_system': dict(),
                                             'energy_revenue': dict(),
                                             'average_energy_price': dict(),
                                             }

        # Summarise model input
        self.case_summary = self.summarise_model_inputs(case_options)

    def summarise_model_inputs(self, case_options):
        """Summary of all parameters used for the given scenario

        Parameters
        ----------
        case_options : dict
            Case parameters

        Returns
        -------
        case_summary : dict
            Summary of case parameters, including a unique ID (based on hashed values
            of model inputs)
        """

        # Parameters used for case
        parameter_values = [case_options[i] for i in sorted(case_options, key=str.lower)]

        # Convert parameters to string
        parameter_values_string = ''.join([str(i) for i in parameter_values])

        # Find sha256 of parameter values - used as a unique identifier for the case
        case_id = hashlib.sha256(parameter_values_string.encode('utf-8')).hexdigest()[:8].upper()

        # Summary of model options, identified by the hash value of these options
        case_summary = {'case_id': case_id, **case_options}

        return case_summary

    def store_scenario_results(self, model_object, calibration_interval, scenario_index):
        """Store results from each scenario

        Parameters
        ----------
        model_object : pyomo model object
            Power flow model object

        calibration_interval : int
            Index of calibration interval being investigated

        scenario_index : int
            Index of scenario being investigated
        """

        # Nodal prices
        self.scenario_nodal_prices[(calibration_interval, scenario_index)] = {n: model_object.model.dual[model_object.model.POWER_BALANCE[n]] for n in model_object.model.OMEGA_N}

        # Power output from each generator
        self.scenario_power_output[(calibration_interval, scenario_index)] = model_object.model.p.get_values()

        # Store scenario metrics
        # ----------------------
        # Net scheme revenue from regulated 'fossil' generators for each scenario [$]
        self.scenario_metrics['net_scheme_revenue_dispatchable_generators'][(calibration_interval, scenario_index)] = model_object.model.NET_SCHEME_REVENUE_DISPATCHABLE_GENERATORS.expr()

        # Net scheme revenue that would need to be paid to intermittent renewables if included in scheme [$]
        self.scenario_metrics['net_scheme_revenue_intermittent_generators'][(calibration_interval, scenario_index)] = model_object.model.NET_SCHEME_REVENUE_INTERMITTENT_GENERATORS.expr()

        # Total emissions [tCO2]
        self.scenario_metrics['total_emissions_tCO2'][(calibration_interval, scenario_index)] = model_object.model.TOTAL_EMISSIONS.expr()

        # Generator energy output
        self.scenario_metrics['dispatchable_generator_energy_MWh'][(calibration_interval, scenario_index)] = {g: model_object.model.p[g].value * model_object.model.BASE_POWER.value * model_object.model.SCENARIO_DURATION.value for g in model_object.model.OMEGA_G}

        # Total emissions from dispatchable generators [tCO2]
        self.scenario_metrics['total_dispatchable_generator_energy_MWh'][(calibration_interval, scenario_index)] = model_object.model.TOTAL_DISPATCHABLE_GENERATOR_ENERGY.expr()

        # Total energy from intermittent generators [MWh]
        self.scenario_metrics['total_intermittent_energy_MWh'][(calibration_interval, scenario_index)] = model_object.model.TOTAL_INTERMITTENT_ENERGY.expr()

        # Total system energy demand [MWh]
        self.scenario_metrics['total_demand_MWh'][(calibration_interval, scenario_index)] = model_object.model.TOTAL_ENERGY_DEMAND.expr()

        # Total revenue from energy sales (nodal price x nodal demand x scenario duration)
        self.scenario_metrics['energy_revenue'][(calibration_interval, scenario_index)] = sum(model_object.model.dual[model_object.model.POWER_BALANCE[n]] * model_object.model.BASE_POWER.value * model_object.model.D[n].value * model_object.model.BASE_POWER.value * model_object.model.SCENARIO_DURATION.value for n in model_object.model.OMEGA_N)

    def store_calibration_interval_metrics(self, model_object, calibration_interval, case_options):
        """Compute aggregate statistics for given calibration interval

        Parameters
        ----------
        model_object : pyomo model object
            Power flow model object

        calibration_interval : int
            Index of calibration interval being investigated

        case_options : dict
            Dictionary describing case parameters
        """

        # Net scheme revenue from regulated 'fossil' generators for each calibration interval [$]
        self.calibration_interval_metrics['net_scheme_revenue_dispatchable_generators'][calibration_interval] = sum(self.scenario_metrics['net_scheme_revenue_dispatchable_generators'][(calibration_interval, s)] for s in model_object.df_scenarios.columns.levels[1])

        # Net scheme revenue that would need to be paid to intermittent renewables each calibration interval if included in scheme [$]
        self.calibration_interval_metrics['net_scheme_revenue_intermittent_generators'][calibration_interval] = sum(self.scenario_metrics['net_scheme_revenue_intermittent_generators'][(calibration_interval, s)] for s in model_object.df_scenarios.columns.levels[1])

        # Emissions intensities for dispatchable generators for the given calibration interval
        self.calibration_interval_metrics['dispatchable_generator_emissions_intensities'][calibration_interval] = {g: model_object.model.E_HAT[g].expr() for g in model_object.model.OMEGA_G}

        # Total emissions [tCO2]
        self.calibration_interval_metrics['total_emissions_tCO2'][calibration_interval] = sum(self.scenario_metrics['total_emissions_tCO2'][(calibration_interval, s)] for s in model_object.df_scenarios.columns.levels[1])

        # Dispatchable generator energy output [MWh]
        self.calibration_interval_metrics['dispatchable_generator_energy_MWh'][calibration_interval] = {g: sum(self.scenario_metrics['dispatchable_generator_energy_MWh'][(calibration_interval, s)][g] for s in model_object.df_scenarios.columns.levels[1]) for g in model_object.model.OMEGA_G}

        # Total output from generators subjected to emissions policy [MWh]
        self.calibration_interval_metrics['total_dispatchable_generator_energy_MWh'][calibration_interval] = sum(self.scenario_metrics['total_dispatchable_generator_energy_MWh'][(calibration_interval, s)] for s in model_object.df_scenarios.columns.levels[1])

        # Total energy from intermittent generators [MWh] (these incumbent generators are generally not eligible for payments)
        self.calibration_interval_metrics['total_intermittent_energy_MWh'][calibration_interval] = sum(self.scenario_metrics['total_intermittent_energy_MWh'][(calibration_interval, s)] for s in model_object.df_scenarios.columns.levels[1])

        # Total energy demand in given calibration interval [MWh]
        self.calibration_interval_metrics['total_demand_MWh'][calibration_interval] = sum(self.scenario_metrics['total_demand_MWh'][(calibration_interval, s)] for s in model_object.df_scenarios.columns.levels[1])

        # Average emissions intensity of all generators (including renewables) [tCO2/MWh]
        self.calibration_interval_metrics['average_emissions_intensity_system'][calibration_interval] = self.calibration_interval_metrics['total_emissions_tCO2'][calibration_interval] / self.calibration_interval_metrics['total_demand_MWh'][calibration_interval]

        # Total revenue from energy sales for given calibration interval [$]
        self.calibration_interval_metrics['energy_revenue'][calibration_interval] = sum(self.scenario_metrics['energy_revenue'][(calibration_interval, s)] for s in model_object.df_scenarios.columns.levels[1])

        # Average energy price [$/MWh]
        self.calibration_interval_metrics['average_energy_price'][calibration_interval] = self.calibration_interval_metrics['energy_revenue'][calibration_interval] / self.calibration_interval_metrics['total_demand_MWh'][calibration_interval]

        # Metrics that depend on whether or not renewables are subject to emissions policy
        if ('renewables_eligibility' in case_options) and case_options.get('renewables_eligibility') == 'eligible':
            # Net scheme revenue when intermittent renewables are covered by policy [$]
            self.calibration_interval_metrics['net_scheme_revenue'][calibration_interval] = self.calibration_interval_metrics['net_scheme_revenue_dispatchable_generators'][calibration_interval] + self.calibration_interval_metrics['net_scheme_revenue_intermittent_generators'][calibration_interval]

            # Average emissions intensity of all generators subject to emissions policy when renewables included [tCO2/MWh]
            self.calibration_interval_metrics['average_emissions_intensity_regulated_generators'][calibration_interval] = self.calibration_interval_metrics['total_emissions_tCO2'][calibration_interval] / (self.calibration_interval_metrics['total_dispatchable_generator_energy_MWh'][calibration_interval] + self.calibration_interval_metrics['total_intermittent_energy_MWh'][calibration_interval])

        else:
            # Net scheme revenue when existing renewables not covered by policy [$]
            self.calibration_interval_metrics['net_scheme_revenue'][calibration_interval] = self.calibration_interval_metrics['net_scheme_revenue_dispatchable_generators'][calibration_interval]

            # Average emissions intensity of all generators subject to emissions policy when renewables not included [tCO2/MWh]
            self.calibration_interval_metrics['average_emissions_intensity_regulated_generators'][calibration_interval] = self.calibration_interval_metrics['total_emissions_tCO2'][calibration_interval] / self.calibration_interval_metrics['total_dispatchable_generator_energy_MWh'][calibration_interval]

        # Record rolling scheme revenue at end of calibration interval [$]
        self.calibration_interval_metrics['rolling_scheme_revenue_interval_end'][calibration_interval] = self.calibration_interval_metrics['rolling_scheme_revenue_interval_start'][calibration_interval] + self.calibration_interval_metrics['net_scheme_revenue'][calibration_interval]

    def save_results(self, output_dir, model_object):
        """Save results

        Parameters
        ----------
        output_dir : str
            Directory for model results files

        model_object : pyomo model object
            Pyomo object used to construct power-flow model
        """

        # Save scenario nodal prices
        with open(f"{output_dir}/{self.case_summary['case_id']}_scenario_nodal_prices.pickle", 'wb') as f:
            pickle.dump(self.scenario_nodal_prices, f)

        # Save scenario power output from individual generators
        with open(f"{output_dir}/{self.case_summary['case_id']}_scenario_power_output.pickle", 'wb') as f:
            pickle.dump(self.scenario_power_output, f)

        # Save scenario metrics
        with open(f"{output_dir}/{self.case_summary['case_id']}_scenario_metrics.pickle", 'wb') as f:
            pickle.dump(self.scenario_metrics, f)

        # Save calibration interval metrics
        with open(f"{output_dir}/{self.case_summary['case_id']}_calibration_interval_metrics.pickle", 'wb') as f:
            pickle.dump(self.calibration_interval_metrics, f)

        # Save case summary
        with open(f"{output_dir}/{self.case_summary['case_id']}_case_summary.pickle", 'wb') as f:
            pickle.dump(self.case_summary, f)

        # Save generators information (note: SRMCs are perturbed when loading data. Good to keep track
        # of generator cost data used in analysis)
        with open(f"{output_dir}/{self.case_summary['case_id']}_generators.pickle", 'wb') as f:
            pickle.dump(model_object.df_g, f)


class Utils:
    """General utilities useful when running model"""

    def __init__(self, case_options):
        """General model utilities

        Parameters
        ----------
        case_options : dict
            Description of case parameters
        """

        # Options describing case
        self.case_options = case_options

    def format_input(self, input_data):
        """Format weekly targets / indicator dictionaries so they can be used in revenue rebalancing or MPC updates

        Parameters
        ----------
        input_data : dict
            Target or indicator data corresponding to calibration intervals

        Returns
        -------
        formatted_input : dict
            Formatted dictionary converting input_data into format that can be consumed by different model components.
            First key is calibration interval, second key is forecast interval e.g. {1 : {1: x, 2: y}}. First key is
            calibration interval, 1st key in inner dict corresponds to first forecast interval, yielding x, 2nd key
            corresponds to second forecast interval, yielding y.
        """

        # Container for formatted input values
        formatted_input = dict()

        # For each calibration interval
        for calibration_interval in range(1, self.case_options.get('model_horizon') + 1 - self.case_options.get('forecast_intervals') + 1):

            # Construct a container for forecast values corresponding to the calibration interval
            formatted_input[calibration_interval] = dict()

            # For each forecast interval
            for forecast_interval in range(1, self.case_options.get('forecast_intervals') + 1):

                # Place input data at correct location corresponding to forecast and calibration interval indices
                formatted_input[calibration_interval][forecast_interval] = input_data[calibration_interval + forecast_interval - 1]

        return formatted_input

    def case_options_valid(self):
        """Check that model options are valid"""

        if 'update_mode' not in self.case_options:
            raise(Exception('Must specify an update mode'))

        if self.case_options.get('update_mode') not in ['NO_UPDATE', 'REVENUE_REBALANCE_UPDATE', 'MPC_UPDATE']:
            raise Warning(f"Unexpected update_mode encountered: {self.case_options.get('update_mode')}")

        if 'shock_option' not in self.case_options:
            raise(Exception('Must specify shock option'))

        if self.case_options.get('shock_option') not in ['NO_SHOCKS', 'EMISSIONS_INTENSITY_SHOCK']:
            raise Warning(f"Unexpected shock_option encountered: {self.case_options.get('shock_option')}")

        if (self.case_options.get('update_mode') == 'MPC_UPDATE') and 'forecast_intervals' not in self.case_options:
            raise(Exception('forecast_intervals not given. Must be specified if using MPC updating.'))

        return True


class ApplyShock(RecordResults):
    """Apply shock to model"""

    def __init__(self, output_dir, case_options):
        """Load methods used to record model results"""
        super().__init__(case_options)

        # Directory for model output files
        self.output_dir = output_dir

        # Options corresponding to case being investigated
        self.case_options = case_options

    def apply_emissions_intensity_shock(self, model_object):
        """Data used to update emissions intensities for generators if shock occurs

        Parameters
        ----------
        model_object : pyomo model object
            Power-flow model object


        Returns
        -------
        model_object : pyomo model object
            Power-flow model object with updated (shocked) generator emissions intensities.
        """

        # Set seed so random shocks can be reproduced
        np.random.seed(self.case_options.get('seed'))

        # Augment original emissions intensity by random scaling factor between 0.8 and 1
        df_emissions_intensity_shock_factor = pd.Series(index=sorted(model_object.model.OMEGA_G),
                                                        data=np.random.uniform(0.8, 1, len(model_object.model.OMEGA_G)))

        # Loop through generators
        for g in model_object.model.OMEGA_G:

            # Augment generator emissions intensities
            model_object.model.EMISSIONS_INTENSITY_SHOCK_FACTOR[g] = float(df_emissions_intensity_shock_factor.loc[g])

        # Save emissions intensity schock factor
        with open(f"{self.output_dir}/{self.case_summary['case_id']}_emissions_intensity_shock_factor.pickle", 'wb') as f:
            pickle.dump(df_emissions_intensity_shock_factor, f)

        return model_object
