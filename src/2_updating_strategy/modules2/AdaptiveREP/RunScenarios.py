import time
import pickle
import hashlib

import numpy as np
import pandas as pd

from .MPC import MPCModel
from .DCOPF import DCOPFModel
# from .model_utils import


def run_scenarios(data_dir, scenarios_dir, output_dir, **kwargs):
    """Run model with given parameters

    Parameters
    ----------
    data_dir : str
        Path to directory containing core data files used in model initialisation

    scenarios_dir : str
        Path to directory containing representative operating scenarios

    output_dir : str
        Path to directory where results files will be written


    Keyword arguments
    -----------------
    shock_option : str
        Specifies type of shock to which system will be subjected. Options

        Options:
            NO_SHOCKS                 - No shocks to system
            EMISSIONS_INTENSITY_SHOCK - Emissions intensity scaled by a random number between 0.8
                                        and 1 at 'week_of_shock'
    update_mode : str
        Specifies how baseline should be updated each week.

        Options:
            NO_UPDATE                -  Same baseline in next iteration
            REVENUE_REBALANCE_UPDATE -  Recalibrate baseline by trying to re-balance net scheme
                                        revenue every interval

    intermittent_generators_regulated : dict or None
        Dictionary containing boolean indicators for each week. If 'True' for a given week, intermittent
        generators are subject to the emissions policy and receive payments under the scheme.
        Default = None.

    forecast_generator_emissions_intensity : dict or None
        Forecast generator emissions intensities for future periods [tCO2/MWh]. Default = None.

    forecast_generator_energy : dict or None
        Forecast generator energy output for future periods [MWh]. Default = None.

    forecast_intermittent_generator_energy : dict or None
        Forecast intermittent generator output for future periods [MWh]. Default = None

    forecast_uncertainty_increment : float or None
        Scaling factor increment used to introduce uncertainty in forecasts. E.g. if 0.05, then
        forecast values for the following week are scaled by a uniformly distributed random
        number in the interval (0.95, 1.05). This scaling factor is incrememented by the same amount
        for future periods. E.g. for the second week the scaling factor would be in the interval (0.9, 1.1)

    forecast_interval_mpc : int or None
        Forecast interval for MPC controller. Default = None.

    forecast_shock : bool or None
        Indicates if emissions intensity shock is anticipated in forecast signals.
        True = shock is anticipated, False = shock is unanticipated. Default = None.

    week_of_shock : int or None
        Index for week at which either generation or emissions intensities shocks will be implemented / begin.
        Default = 10.

    default_baseline : float
        Initialised emissions intensity baseline value [tCO2/MWh]. Default = 1 tCO2/MWh.

    initial_permit_price : float
        Initialised permit price value [$/tCO2]. Default = 40 $/tCO2.

    initial_rolling_scheme_revenue : float
        Initialised rolling scheme revenue value [$]. Default = $0.

    target_scheme_revenue : dict or None
        Net scheme revenue target defined for each period [$]. Default = None.

    seed : int
        Seed used for random number generator. Allows shocks to be reproduced. Default = 10.

    model_horizon : int
        Total number of weeks to investigate. Default is 52 weeks (whole year).


    Returns
    -------
    run_id : str
        ID used to identify model run
    """

    # Print model being run
    print(f'Running model: {kwargs}')

    # Summary of all parameters used for the given scenario
    parameter_values = [kwargs[i] for i in sorted(kwargs, key=str.lower)]

    # Convert parameters to string
    parameter_values_string = ''.join([str(i) for i in parameter_values])

    # Find sha256 of parameter values - used as a unique identifier for the model run
    run_id = hashlib.sha256(parameter_values_string.encode('utf-8')).hexdigest()[:8].upper()

    # Summary of model options, identified by the hash value of these options
    run_summary = {run_id: kwargs}

    # Check if model parameters valid
    # -------------------------------
    if 'update_mode' not in kwargs:
        raise(Exception('Must specify update mode'))

    if kwargs.get('update_mode') not in ['NO_UPDATE', 'REVENUE_REBALANCE_UPDATE', 'MPC_UPDATE']:
        raise Warning(f"Unexpected update_mode encountered: {kwargs.get('update_mode')}")

    if 'shock_option' not in kwargs:
        raise(Exception('Must specify shock option'))

    if kwargs.get('shock_option') not in ['NO_SHOCKS', 'EMISSIONS_INTENSITY_SHOCK']:
        raise Warning(f"Unexpected shock_option encountered: {kwargs.get('shock_option')}")

    # Create model objects
    # --------------------
    # Instantiate DCOPF model object
    DCOPF = DCOPFModel(data_dir=data_dir, scenarios_dir=scenarios_dir)

    # Instantiate Model Predictive Controller if MPC update specified
    if kwargs.get('update_mode') == 'MPC_UPDATE':
        if 'forecast_interval_mpc' not in kwargs:
            raise(Exception('forecast_interval_mpc not given. Must specify if using MPC updating.'))
        MPC = MPCModel(generator_index=DCOPF.model.OMEGA_G, forecast_interval=kwargs.get('forecast_interval_mpc'))

    # Result containers
    # -----------------
    # Prices at each node for each scenario
    scenario_nodal_prices = dict()

    # Power output for each scenario
    scenario_power_output = dict()

    # Results for each scenario
    scenario_metrics = {'net_scheme_revenue_dispatchable_generators': dict(),
                        'net_scheme_revenue_intermittent_generators': dict(),
                        'total_emissions_tCO2': dict(),
                        'dispatchable_generator_energy_MWh': dict(),
                        'total_dispatchable_generator_energy_MWh': dict(),
                        'total_intermittent_energy_MWh': dict(),
                        'total_demand_MWh': dict(),
                        'energy_revenue': dict(),
                        }

    # Aggregated results for each week
    week_metrics = {'baseline': dict(),
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

    # Random shocks
    # -------------
    # Set seed so random shocks can be reproduced if needed
    np.random.seed(kwargs.get('seed'))

    # Augment original emissions intensity by random scaling factor between 0.8 and 1
    if kwargs.get('shock_option') == 'EMISSIONS_INTENSITY_SHOCK':
        df_emissions_intensity_shock_factor = pd.Series(index=sorted(DCOPF.model.OMEGA_G), data=np.random.uniform(0.8, 1, len(DCOPF.model.OMEGA_G)))

    # Run scenarios
    # -------------
    # Week indices for which model will be run (can run for less than one year by using model_horizon)
    weeks = DCOPF.df_scenarios.columns.levels[0][:kwargs.get('model_horizon')]

    # For each week
    for week_index in weeks:
        # Start clock to see how long it takes to solve all scenarios for each week
        t0 = time.time()

        # Initialise policy parameters if the first week
        if week_index == 1:
            # Initialise permit price
            permit_price = kwargs.get('initial_permit_price')

            # Initialise rolling scheme revenue (value at the end of previous calibration interval)
            rolling_scheme_revenue_interval_start = kwargs.get('initial_rolling_scheme_revenue')

        # Update rolling scheme revenue (amount of money in bank account at start of calibration interval)
        if week_index > 1:
            rolling_scheme_revenue_interval_start = week_metrics['rolling_scheme_revenue_interval_end'][week_index - 1]

        # Record amount of money in bank account at start of calibration interval
        week_metrics['rolling_scheme_revenue_interval_start'][week_index] = rolling_scheme_revenue_interval_start

        # Compute baseline
        # ----------------
        if kwargs.get('update_mode') == 'NO_UPDATE':
            # No update to baseline (should be 0 tCO2/MWh)
            baseline = kwargs.get('default_baseline')

            # Assume renewables not included
            renewables_included_this_week = False

        elif kwargs.get('update_mode') == 'REVENUE_REBALANCE_UPDATE':

            # Forecast regulated generator emissions in next period
            forecast_regulated_generator_total_emissions = sum(kwargs.get('forecast_generator_emissions_intensity')[week_index][1][g] * kwargs.get('forecast_generator_energy')[week_index][1][g] for g in DCOPF.model.OMEGA_G)

            # Forecast energy output from 'fossil' generators (definitely under policy's remit)
            forecast_fossil_generator_total_energy = sum(kwargs.get('forecast_generator_energy')[week_index][1][g] for g in DCOPF.model.OMEGA_G)

            # Forecast energy from intermittent reneweable generators (may be under scheme's remit)
            forecast_intermittent_generator_total_energy = kwargs.get('forecast_intermittent_generator_energy')[week_index][1]

            # Forecast regulated generator energy in next period. Value may change depending on whether or not
            # intermittent generators are subject to the scheme's remit.

            # Indicates if renewables are eligible for payments (default=False)
            if 'intermittent_generators_regulated' in kwargs:
                renewables_included_this_week = kwargs.get('intermittent_generators_regulated')[week_index][1]
            else:
                renewables_included_this_week = False

            if renewables_included_this_week:
                # Intermittent generators are part of scheme's remit, and receive payments
                forecast_regulated_generator_total_energy += forecast_fossil_generator_total_energy + forecast_intermittent_generator_total_energy
            else:
                # Only output from fossil generators considered
                forecast_regulated_generator_total_energy = forecast_fossil_generator_total_energy

            # Forecast regulated generator average emissions intensity
            forecast_regulated_generator_average_emissions_intensity = forecast_regulated_generator_total_emissions / forecast_regulated_generator_total_energy

            # Update baseline seeking to re-balance net scheme revenue every period based on forecast output
            baseline = forecast_regulated_generator_average_emissions_intensity - ((kwargs.get('target_scheme_revenue')[week_index][1] - rolling_scheme_revenue_interval_start) / (permit_price * forecast_regulated_generator_total_energy))

            # Set baseline to 0 if updated value less than zero
            if baseline < 0:
                baseline = 0

        elif kwargs.get('update_mode') == 'MPC_UPDATE':
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

        # Record baseline applying for the given week
        week_metrics['baseline'][week_index] = baseline

        # Apply emissions intensity shock if shock option specified. Shock occurs once at week_index
        # given by week_of_shock
        if (kwargs.get('shock_option') == 'EMISSIONS_INTENSITY_SHOCK') and (week_index == kwargs.get('week_of_shock')):
            # Loop through generators
            for g in DCOPF.model.OMEGA_G:
                # Augment generator emissions intensity factor
                DCOPF.model.EMISSIONS_INTENSITY_SHOCK_FACTOR[g] = float(df_emissions_intensity_shock_factor.loc[g])

        # For each representative scenario approximating a week's operating state
        for scenario_index in DCOPF.df_scenarios.columns.levels[1]:
            # Update model parameters
            DCOPF.update_model_parameters(week_index=week_index,
                                          scenario_index=scenario_index,
                                          baseline=baseline,
                                          permit_price=permit_price)

            # Solve model
            DCOPF.solve_model()

            # Store results
            # -------------
            # Nodal prices
            scenario_nodal_prices[(week_index, scenario_index)] = {n: DCOPF.model.dual[DCOPF.model.POWER_BALANCE[n]] for n in DCOPF.model.OMEGA_N}

            # Power output from each generator
            scenario_power_output[(week_index, scenario_index)] = DCOPF.model.p.get_values()

            # Store scenario metrics
            # ----------------------
            # Net scheme revenue from regulated 'fossil' generators for each scenario [$]
            scenario_metrics['net_scheme_revenue_dispatchable_generators'][(week_index, scenario_index)] = DCOPF.model.NET_SCHEME_REVENUE_DISPATCHABLE_GENERATORS.expr()

            # Net scheme revenue that would need to be paid to intermittent renewables if included in scheme [$]
            scenario_metrics['net_scheme_revenue_intermittent_generators'][(week_index, scenario_index)] = DCOPF.model.NET_SCHEME_REVENUE_INTERMITTENT_GENERATORS.expr()

            # Total emissions [tCO2]
            scenario_metrics['total_emissions_tCO2'][(week_index, scenario_index)] = DCOPF.model.TOTAL_EMISSIONS.expr()

            # Generator energy output
            scenario_metrics['dispatchable_generator_energy_MWh'][(week_index, scenario_index)] = {g: DCOPF.model.p[g].value * DCOPF.model.BASE_POWER.value * DCOPF.model.SCENARIO_DURATION.value for g in DCOPF.model.OMEGA_G}

            # Total emissions from dispatchable generators [tCO2]
            scenario_metrics['total_dispatchable_generator_energy_MWh'][(week_index, scenario_index)] = DCOPF.model.TOTAL_DISPATCHABLE_GENERATOR_ENERGY.expr()

            # Total energy from intermittent generators [MWh]
            scenario_metrics['total_intermittent_energy_MWh'][(week_index, scenario_index)] = DCOPF.model.TOTAL_INTERMITTENT_ENERGY.expr()

            # Total system energy demand [MWh]
            scenario_metrics['total_demand_MWh'][(week_index, scenario_index)] = DCOPF.model.TOTAL_ENERGY_DEMAND.expr()

            # Total revenue from energy sales (nodal price x nodal demand x scenario duration)
            scenario_metrics['energy_revenue'][(week_index, scenario_index)] = sum(DCOPF.model.dual[DCOPF.model.POWER_BALANCE[n]] * DCOPF.model.BASE_POWER.value * DCOPF.model.D[n].value * DCOPF.model.BASE_POWER.value * DCOPF.model.SCENARIO_DURATION.value for n in DCOPF.model.OMEGA_N)

        # Compute aggregate statistics for given week
        # -------------------------------------------
        # Net scheme revenue from regulated 'fossil' generators for each week [$]
        week_metrics['net_scheme_revenue_dispatchable_generators'][week_index] = sum(scenario_metrics['net_scheme_revenue_dispatchable_generators'][(week_index, s)] for s in DCOPF.df_scenarios.columns.levels[1])

        # Net scheme revenue that would need to be paid to intermittent renewables each week if included in scheme [$]
        week_metrics['net_scheme_revenue_intermittent_generators'][week_index] = sum(scenario_metrics['net_scheme_revenue_intermittent_generators'][(week_index, s)] for s in DCOPF.df_scenarios.columns.levels[1])

        # Emissions intensities for regulated generators for the given week
        week_metrics['dispatchable_generator_emissions_intensities'][week_index] = {g: DCOPF.model.E_HAT[g].expr() for g in DCOPF.model.OMEGA_G}

        # Total emissions [tCO2]
        week_metrics['total_emissions_tCO2'][week_index] = sum(scenario_metrics['total_emissions_tCO2'][(week_index, s)] for s in DCOPF.df_scenarios.columns.levels[1])

        # Weekly energy output for each generator
        week_metrics['dispatchable_generator_energy_MWh'][week_index] = {g: sum(scenario_metrics['dispatchable_generator_energy_MWh'][(week_index, s)][g] for s in DCOPF.df_scenarios.columns.levels[1]) for g in DCOPF.model.OMEGA_G}

        # Total output from generators subjected to emissions intensity policy [MWh]
        week_metrics['total_dispatchable_generator_energy_MWh'][week_index] = sum(scenario_metrics['total_dispatchable_generator_energy_MWh'][(week_index, s)] for s in DCOPF.df_scenarios.columns.levels[1])

        # Total energy from intermittent generators [MWh] (these incumbent generators are generally not subjected to policy)
        week_metrics['total_intermittent_energy_MWh'][week_index] = sum(scenario_metrics['total_intermittent_energy_MWh'][(week_index, s)] for s in DCOPF.df_scenarios.columns.levels[1])

        # Total energy demand in given week [MWh]
        week_metrics['total_demand_MWh'][week_index] = sum(scenario_metrics['total_demand_MWh'][(week_index, s)] for s in DCOPF.df_scenarios.columns.levels[1])

        # Average emissions intensity of whole system (including renewables) [tCO2/MWh]
        week_metrics['average_emissions_intensity_system'][week_index] = week_metrics['total_emissions_tCO2'][week_index] / week_metrics['total_demand_MWh'][week_index]

        # Total revenue from energy sales for given week [$]
        week_metrics['energy_revenue'][week_index] = sum(scenario_metrics['energy_revenue'][(week_index, s)] for s in DCOPF.df_scenarios.columns.levels[1])

        # Average energy price [$/MWh]
        week_metrics['average_energy_price'][week_index] = week_metrics['energy_revenue'][week_index] / week_metrics['total_demand_MWh'][week_index]

        # Metrics that depend on whether or not renewables are subject to emissions policy
        if renewables_included_this_week:
            # Net scheme revenue when intermittent renewables are covered by policy [$]
            week_metrics['net_scheme_revenue'][week_index] = week_metrics['net_scheme_revenue_dispatchable_generators'][week_index] + week_metrics['net_scheme_revenue_intermittent_generators'][week_index]

            # Average emissions intensity of all generators subject to emissions policy generators when renewables included [tCO2/MWh]
            week_metrics['average_emissions_intensity_regulated_generators'][week_index] = week_metrics['total_emissions_tCO2'][week_index] / (week_metrics['total_dispatchable_generator_energy_MWh'][week_index] + week_metrics['total_intermittent_energy_MWh'][week_index])

        else:
            # Net scheme revenue when existing renewables not covered by policy [$]
            week_metrics['net_scheme_revenue'][week_index] = week_metrics['net_scheme_revenue_dispatchable_generators'][week_index]

            # Average emissions intensity of all generators subject to emissions policy generators when renewables included [tCO2/MWh]
            week_metrics['average_emissions_intensity_regulated_generators'][week_index] = week_metrics['total_emissions_tCO2'][week_index] / week_metrics['total_dispatchable_generator_energy_MWh'][week_index]

        # Record rolling scheme revenue at end of calibration interval [$]
        week_metrics['rolling_scheme_revenue_interval_end'][week_index] = rolling_scheme_revenue_interval_start + week_metrics['net_scheme_revenue'][week_index]

        print(f'Completed week {week_index} in {time.time()-t0:.2f}s')

    # Save results
    # ------------
    with open(f'{output_dir}/{run_id}_scenario_nodal_prices.pickle', 'wb') as f:
        pickle.dump(scenario_nodal_prices, f)

    with open(f'{output_dir}/{run_id}_scenario_power_output.pickle', 'wb') as f:
        pickle.dump(scenario_power_output, f)

    with open(f'{output_dir}/{run_id}_scenario_metrics.pickle', 'wb') as f:
        pickle.dump(scenario_metrics, f)

    with open(f'{output_dir}/{run_id}_week_metrics.pickle', 'wb') as f:
        pickle.dump(week_metrics, f)

    with open(f'{output_dir}/{run_id}_run_summary.pickle', 'wb') as f:
        pickle.dump(run_summary, f)

    with open(f'{output_dir}/{run_id}_generators.pickle', 'wb') as f:
        pickle.dump(DCOPF.df_g, f)

    if kwargs.get('shock_option') == 'EMISSIONS_INTENSITY_SHOCK':
        with open(f'{output_dir}/{run_id}_emissions_intensity_shock_factor.pickle', 'wb') as f:
            pickle.dump(df_emissions_intensity_shock_factor, f)

    return run_id
