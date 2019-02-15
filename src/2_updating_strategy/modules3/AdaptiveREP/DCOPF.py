from math import pi

import numpy as np
from pyomo.environ import *

from .ModelData import OrganisedData


class DCOPFModel(OrganisedData):
    "Create DCOPF model"

    def __init__(self, data_dir, scenarios_dir):
        # Load model data
        super().__init__(data_dir, scenarios_dir)

        # Initialise DCOPF model
        self.model = self.create_model()

        # Setup solver
        # ------------
        # Import dual variables
        self.model.dual = Suffix(direction=Suffix.IMPORT)

        # Specify solver to be used and output format
        self.opt = SolverFactory('gurobi', solver_io='mps')

        # Parameters used for different scenarios
        # ---------------------------------------
        # Calibration interval index
        self.calibration_interval = None

        # Scenario index
        self.scenario_index = None

    def create_model(self):
        """Create model object

        Note: Normalising model by scaling values related to power by 100 MVA
        (essentially using p.u. units). Also scaling prices by same factor (100)
        to improve numerical stability (trying to keep short-run marginal cost values
        between 0 and 1). Need to re-scale when analysing results.
        """

        # Initialise model
        model = ConcreteModel()

        # Sets
        # ----
        # Nodes
        model.OMEGA_N = Set(initialize=self.df_n.index)

        # Generators
        model.OMEGA_G = Set(initialize=self.get_all_dispatchable_fossil_generator_duids())

        # AC edges
        ac_edges = self.get_all_ac_edges()
        model.OMEGA_NM = Set(initialize=ac_edges)

        # Sets of branches for which aggregate AC interconnector limits are defined
        ac_limits = self.get_ac_interconnector_summary()
        model.OMEGA_J = Set(initialize=ac_limits.index)

        # HVDC links
        model.OMEGA_H = Set(initialize=self.df_hvdc_links.index)

        # Parameters
        # ----------
        # System base power
        model.BASE_POWER = Param(initialize=100)

        # Emissions intensity baseline
        model.PHI = Param(initialize=0, mutable=True)

        # Permit price
        model.TAU = Param(initialize=0, mutable=True)

        # Generator emissions intensities
        def E_RULE(model, g):
            return float(self.df_g.loc[g, 'EMISSIONS'])
        model.E = Param(model.OMEGA_G, rule=E_RULE)

        # Admittance matrix
        admittance_matrix = self.get_admittance_matrix()

        def B_RULE(model, n, m):
            return float(np.imag(admittance_matrix.loc[n, m]))
        model.B = Param(model.OMEGA_NM, rule=B_RULE)

        # Reference nodes
        reference_nodes = self.get_reference_nodes()

        def S_RULE(model, n):
            if n in reference_nodes:
                return 1
            else:
                return 0
        model.S = Param(model.OMEGA_N, rule=S_RULE)

        # Generator short-run marginal costs
        def C_RULE(model, g):
            marginal_cost = float(self.df_g.loc[g, 'SRMC_2016-17'])
            return marginal_cost / model.BASE_POWER
        model.C = Param(model.OMEGA_G, rule=C_RULE)

        # Demand
        model.D = Param(model.OMEGA_N, initialize=0, mutable=True)

        # Max voltage angle difference between connected nodes
        model.THETA_DELTA = Param(initialize=float(pi / 2))

        # HVDC incidence matrix
        hvdc_incidence_matrix = self.get_HVDC_incidence_matrix()

        def K_RULE(model, n, h):
            return float(hvdc_incidence_matrix.loc[n, h])
        model.K = Param(model.OMEGA_N, model.OMEGA_H, rule=K_RULE)

        # Aggregate AC interconnector flow limits
        def F_RULE(model, j):
            power_flow_limit = float(ac_limits.loc[j, 'LIMIT'])
            return power_flow_limit / model.BASE_POWER
        model.F = Param(model.OMEGA_J, rule=F_RULE)

        # Power injections from intermittent generators
        model.INTERMITTENT = Param(model.OMEGA_N, initialize=0, mutable=True)

        # Power injections from hydro generators
        model.HYDRO = Param(model.OMEGA_N, initialize=0, mutable=True)

        # Fixed power injections
        def R_RULE(model, n):
            return model.INTERMITTENT[n] + model.HYDRO[n]
        model.R = Expression(model.OMEGA_N, rule=R_RULE)

        # Maximum power output
        def REGISTERED_CAPACITY_RULE(model, g):
            registered_capacity = float(self.df_g.loc[g, 'REG_CAP'])
            return registered_capacity / model.BASE_POWER
        model.REGISTERED_CAPACITY = Param(model.OMEGA_G, rule=REGISTERED_CAPACITY_RULE)

        # Emissions intensity shock indicator parameter. Used to scale original emissions intensities.
        model.EMISSIONS_INTENSITY_SHOCK_FACTOR = Param(model.OMEGA_G, initialize=1, mutable=True)

        # Duration of each scenario (will be updated each time model is run. Useful when computing
        # total emissions / total scheme revenue etc.)
        model.SCENARIO_DURATION = Param(initialize=0, mutable=True)

        # Variables
        # ---------
        # Generator output (constrained to non-negative values)
        model.p = Var(model.OMEGA_G, within=NonNegativeReals)

        # HVDC flows
        def P_H_RULE(model, h):
            forward_flow_limit = float(self.df_hvdc_links.loc[h, 'FORWARD_LIMIT_MW'])
            reverse_flow_limit = float(self.df_hvdc_links.loc[h, 'REVERSE_LIMIT_MW'])
            return (- reverse_flow_limit / model.BASE_POWER, forward_flow_limit / model.BASE_POWER)
        model.p_H = Var(model.OMEGA_H, bounds=P_H_RULE)

        # Node voltage angles
        model.theta = Var(model.OMEGA_N)

        # Constraints
        # -----------
        # Power balance
        generator_node_map = self.get_generator_node_map(model.OMEGA_G)

        # Dictionary defining relationships between nodes
        # (given a node as the key, the value will be all nodes directly connected to it)
        network_graph = self.get_network_graph()

        # Power balance at each node
        def POWER_BALANCE_RULE(model, n):
            return (- model.D[n]
                    + model.R[n]
                    + sum(model.p[g] for g in generator_node_map[n])
                    - sum(model.B[n, m] * (model.theta[n] - model.theta[m]) for m in network_graph[n])
                    - sum(model.K[n, h] * model.p_H[h] for h in model.OMEGA_H) == 0)
        model.POWER_BALANCE = Constraint(model.OMEGA_N, rule=POWER_BALANCE_RULE)

        # Max power output
        def P_MAX_RULE(model, g):
            return model.p[g] <= model.REGISTERED_CAPACITY[g]
        model.P_MAX = Constraint(model.OMEGA_G, rule=P_MAX_RULE)

        # Reference angle
        def REFERENCE_ANGLE_RULE(model, n):
            if model.S[n] == 1:
                return model.theta[n] == 0
            else:
                return Constraint.Skip
        model.REFERENCE_ANGLE = Constraint(model.OMEGA_N, rule=REFERENCE_ANGLE_RULE)

        # Voltage angle difference constraint
        def VOLTAGE_ANGLE_DIFFERENCE_RULE(model, n, m):
            return model.theta[n] - model.theta[m] - model.THETA_DELTA <= 0
        model.VOLTAGE_ANGLE_DIFFERENCE = Constraint(model.OMEGA_NM, rule=VOLTAGE_ANGLE_DIFFERENCE_RULE)

        # AC interconnector flow constraints
        def AC_FLOW_RULE(model, j):
            return sum(model.B[n, m] * (model.theta[n] - model.theta[m]) for n, m in ac_limits.loc[j, 'branches'])
        model.AC_FLOW = Expression(model.OMEGA_J, rule=AC_FLOW_RULE)

        # Limits on flows over individual branches
        def AC_POWER_FLOW_LIMIT_RULE(model, j):
            return model.AC_FLOW[j] - model.F[j] <= 0
        model.AC_POWER_FLOW_LIMIT = Constraint(model.OMEGA_J, rule=AC_POWER_FLOW_LIMIT_RULE)

        # Expressions
        # -----------
        # Effective emissions intensities (original emissions intensity x scaling factor)
        def E_HAT_RULE(model, g):
            return model.E[g] * model.EMISSIONS_INTENSITY_SHOCK_FACTOR[g]
        model.E_HAT = Expression(model.OMEGA_G, rule=E_HAT_RULE)

        # Total emissions [tCO2]
        model.TOTAL_EMISSIONS = Expression(expr=sum(model.p[g] * model.BASE_POWER * model.E_HAT[g] * model.SCENARIO_DURATION for g in model.OMEGA_G))

        # Total energy demand [MWh]
        model.TOTAL_ENERGY_DEMAND = Expression(expr=sum(model.D[n] * model.BASE_POWER * model.SCENARIO_DURATION for n in model.OMEGA_N))

        # Total energy output from generators eligble for emissions payments [MWh]
        model.TOTAL_DISPATCHABLE_GENERATOR_ENERGY = Expression(expr=sum(model.p[g] * model.BASE_POWER * model.SCENARIO_DURATION for g in model.OMEGA_G))

        # Total energy from intermittent generators [MWh]
        model.TOTAL_INTERMITTENT_ENERGY = Expression(expr=sum(model.INTERMITTENT[n] * model.BASE_POWER * model.SCENARIO_DURATION for n in model.OMEGA_N))

        # Average emissions intensity of regulated generators [tCO2/MWh]
        model.AVERAGE_EMISSIONS_INTENSITY_DISPATCHABLE_GENERATORS = Expression(expr=model.TOTAL_EMISSIONS / model.TOTAL_DISPATCHABLE_GENERATOR_ENERGY)

        # Average emissions intensity of system [tCO2/MWh]
        model.AVERAGE_EMISSIONS_INTENSITY_SYSTEM = Expression(expr=model.TOTAL_EMISSIONS / model.TOTAL_ENERGY_DEMAND)

        # Net scheme revenue from dispatchable 'fossil' generators [$]
        model.NET_SCHEME_REVENUE_DISPATCHABLE_GENERATORS = Expression(expr=sum((model.E_HAT[g] - model.PHI) * model.p[g] * model.BASE_POWER * model.SCENARIO_DURATION * model.TAU * model.BASE_POWER for g in model.OMEGA_G))

        # Net scheme revenue that would need to be paid to intermittent renewables if included in scheme [$]
        model.NET_SCHEME_REVENUE_INTERMITTENT_GENERATORS = Expression(expr=- model.PHI * model.TOTAL_INTERMITTENT_ENERGY * model.TAU * model.BASE_POWER)

        # Objective
        # ---------
        # Cost minimisation
        model.OBJECTIVE = Objective(expr=sum((model.C[g] + ((model.E_HAT[g] - model.PHI) * model.TAU)) * model.p[g] for g in model.OMEGA_G))

        return model

    def update_model_parameters(self, calibration_interval, scenario_index, baseline, permit_price):
        """ Update DCOPF model parameters

        Parameters
        ----------
        calibration_interval : int
            Index of calibration interval for which model is to be run

        scenario_index : int
            Index of scenario that describes operating condition for the given calibration interval

        baseline: float
            Emissions intensity baseline [tCO2/MWh]

        permit price : float
            Permit price [$/tCO2]
        """

        # Update fixed nodal power injections
        for n in self.model.OMEGA_N:
            # Update demand at each node
            self.model.D[n] = float(self.df_scenarios.loc[('demand', n), (calibration_interval, scenario_index)] / self.model.BASE_POWER.value)

            # Update fixed injections from hydro generators
            self.model.HYDRO[n] = float(self.df_scenarios.loc[('hydro', n), (calibration_interval, scenario_index)] / self.model.BASE_POWER.value)

            # Update fixed injections from intermittent renewables
            self.model.INTERMITTENT[n] = float(self.df_scenarios.loc[('intermittent', n), (calibration_interval, scenario_index)] / self.model.BASE_POWER.value)

        # Update emissions intensity baseline [tCO2/MWh]
        self.model.PHI = float(baseline)

        # Update permit price [$/tCO2]
        self.model.TAU = float(permit_price / self.model.BASE_POWER.value)

        # Scenario duration [hrs]
        self.model.SCENARIO_DURATION = float(self.df_scenarios.loc[('hours', 'duration'), (calibration_interval, scenario_index)])

        # Update calibration interval index
        self.calibration_interval = calibration_interval

        # Update scenario index
        self.scenario_index = scenario_index

    def solve_model(self):
        "Solve model"

        self.opt.solve(self.model)
