"""Model Predictive Controller used to update emissions intensity baseline"""

from collections import OrderedDict

from pyomo.environ import *


class MPCModel:
    """Model Predictive Controller used to update emissions intensity baseline"""

    def __init__(self, generator_index, forecast_intervals):
        """Initialise MPC controller parameters

        Parameters
        ----------
        generator_index : list
            Dispatchable geneators in power-flow model

        forecast_intervals : int
            Forecast horizon for MPC controller
        """

        # Create model object
        self.model = self.mpc_baseline(generator_index=generator_index, forecast_intervals=forecast_intervals)

        # Define solver
        self.opt = SolverFactory('gurobi', solver_io='lp')

    def mpc_baseline(self, generator_index, forecast_intervals):
        """Compute baseline path using model predictive control

        Parameters
        ----------
        generator_index : list
            DUIDs for each generator regulated by the emissions intensity scheme

        forecast_intervals : int
            Number of periods over which the MPC controller has to achieve its objective


        Returns
        -------
        model : pyomo model object
            Quadratic program used to find path of emissions intensity baseline that
            achieves a revenue target while minimising changes to the emissions intensity baseline
            over the horizon in which this re-balancing takes place.
        """

        # Initialise model object
        model = ConcreteModel()

        # Sets
        # ----
        # Generators
        model.OMEGA_G = Set(initialize=generator_index)

        # Time index
        model.OMEGA_T = Set(initialize=range(1, forecast_intervals + 1), ordered=True)

        # Parameters
        # ----------
        # Predicted generator emissions intensity for future periods
        model.EMISSIONS_INTENSITY_FORECAST = Param(model.OMEGA_G, model.OMEGA_T, initialize=0, mutable=True)

        # Predicted weekly energy output
        model.ENERGY_FORECAST = Param(model.OMEGA_G, model.OMEGA_T, initialize=0, mutable=True)

        # Predicted weekly energy output from intermittent generators
        model.INTERMITTENT_ENERGY_FORECAST = Param(model.OMEGA_T, initialize=0, mutable=True)

        # Permit price
        model.PERMIT_PRICE = Param(initialize=0, mutable=True)

        # Emissions intensity baseline from previous period
        model.BASELINE_INTERVAL_START = Param(initialize=0, mutable=True)

        # Initial rolling scheme revenue
        model.SCHEME_REVENUE_INTERVAL_START = Param(initialize=0, mutable=True)

        # Rolling scheme revenue target at end of finite horizon
        model.TARGET_SCHEME_REVENUE = Param(initialize=0, mutable=True)

        # Indicator denoting whether energy from intermittent renewables is eligible for scheme payments
        model.RENEWABLES_ELIGIBILITY_INDICATOR = Param(model.OMEGA_T, initialize=0, mutable=True)

        # Variables
        # ---------
        # Emissions intensity baseline
        model.phi = Var(model.OMEGA_T)

        # Constraints
        # -----------
        # Scheme revenue must be at target by end of model horizon
        model.SCHEME_REVENUE = Constraint(expr=model.SCHEME_REVENUE_INTERVAL_START
                                          + sum((model.EMISSIONS_INTENSITY_FORECAST[g, t] - model.phi[t]) * model.ENERGY_FORECAST[g, t] * model.PERMIT_PRICE for g in model.OMEGA_G for t in model.OMEGA_T)
                                          - sum(model.RENEWABLES_ELIGIBILITY_INDICATOR[t] * model.phi[t] * model.INTERMITTENT_ENERGY_FORECAST[t] * model.PERMIT_PRICE for t in model.OMEGA_T)
                                          == model.TARGET_SCHEME_REVENUE)

        # Baseline must be non-negative
        def BASELINE_NONNEGATIVE_RULE(model, t):
            return model.phi[t] >= 0
        model.BASELINE_NONNEGATIVE = Constraint(model.OMEGA_T, rule=BASELINE_NONNEGATIVE_RULE)

        # Objective function
        # ------------------
        # Minimise changes to baseline over finite time horizon
        model.OBJECTIVE = Objective(expr=(((model.phi[model.OMEGA_T.first()] - model.BASELINE_INTERVAL_START) * (model.phi[model.OMEGA_T.first()] - model.BASELINE_INTERVAL_START))
                                          + sum((model.phi[t] - model.phi[t - 1]) * (model.phi[t] - model.phi[t - 1]) for t in model.OMEGA_T if t > model.OMEGA_T.first()))
                                    )
        return model

    def update_model_parameters(self, forecast_emissions_intensities, forecast_generator_energy, forecast_intermittent_energy, renewables_eligibility, permit_price, baseline_interval_start, scheme_revenue_interval_start, target_scheme_revenue):
        """Update parameters used as inputs for the MPC controller

        Parameters
        ----------
        forecast_emissions_intensity : dict
            Expected generator emissions intensities over forecast interval

        forecast_generator_energy : dict
            Forecast weekly energy output from dispatchable generators over the forecast interval

        forecast_intermittent_energy : dict
            Forecast weekly intermittent energy output from generators over the forecast interval

        renewables_eligibility : dict
            Indicator denoting if energy from renewables receives payments under the scheme

        permit_price : float
            Emissions price [tCO2/MWh]

        baseline_interval_start : float
            Emissions intensity baseline implemented for preceding week [tCO2/MWh]

        scheme_revenue_interval_start : float
            Rolling scheme revenue at end of preceding week [$]

        target_scheme_revenue : float
            Target scheme revenue to be obtained in the future [$]
        """

        # For each time interval in the forecast horizon
        for t in self.model.OMEGA_T:

            # If renewbales are eligible for given interval, update indicator parameter
            if renewables_eligibility[t]:
                # Renewables are eligible to receive payments for given interval
                self.model.RENEWABLES_ELIGIBILITY_INDICATOR[t] = float(1)
            else:
                # Renewables are ineligble to receive payments for given interval
                self.model.RENEWABLES_ELIGIBILITY_INDICATOR[t] = float(0)

            # For each generator
            for g in self.model.OMEGA_G:
                # Predicted generator emissions intensities for future periods
                self.model.EMISSIONS_INTENSITY_FORECAST[g, t] = float(forecast_emissions_intensities[t][g])

                # Predicted weekly energy output
                self.model.ENERGY_FORECAST[g, t] = float(forecast_generator_energy[t][g])

                # Predicted weekly energy from intermittent generators
                self.model.INTERMITTENT_ENERGY_FORECAST[t] = float(forecast_intermittent_energy[t])

        # Permit price
        self.model.PERMIT_PRICE = float(permit_price)

        # Emissions intensity baseline from previous period
        self.model.BASELINE_INTERVAL_START = float(baseline_interval_start)

        # Initial rolling scheme revenue at start of interval
        self.model.SCHEME_REVENUE_INTERVAL_START = float(scheme_revenue_interval_start)

        # Target rolling scheme revenue at end of forecast horizon
        self.model.TARGET_SCHEME_REVENUE = float(target_scheme_revenue)

    def solve_model(self):
        """Solve for optimal emissions intensity baseline path"""

        self.opt.solve(self.model)

    def get_optimal_baseline_path(self):
        """Get optimal emissions intenstiy baseline path based on MPC controller"""

        # Optimal emissions intensity baseline path as determined by MPC controller
        optimal_baseline_path = OrderedDict(self.model.phi.get_values())

        return optimal_baseline_path

    def get_next_baseline(self):
        """Get next baseline to be implemented for the coming week"""

        # Optimal path of baselines to be implemented over the finite horizon
        optimal_baseline_path = self.get_optimal_baseline_path()

        # Next 'optimal' emissions intensity baseline to implemented for the coming interval
        next_baseline = float(optimal_baseline_path[self.model.OMEGA_T.first()])

        return next_baseline
