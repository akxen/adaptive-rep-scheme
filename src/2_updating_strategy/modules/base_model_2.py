
# coding: utf-8

# In[1]:


import os
import pickle
from math import pi

import numpy as np
import pandas as pd
from pyomo.environ import *


class RawData:
    "Load raw data to be used in model"
    
    def __init__(self, data_dir, scenarios_dir, seed=10):
        
        # Paths to directories
        # --------------------
        self.data_dir = data_dir
        
        
        # Network data
        # ------------
        # Nodes
        self.df_n = pd.read_csv(os.path.join(self.data_dir, 'egrimod-nem-dataset-v1.3', 'akxen-egrimod-nem-dataset-4806603', 'network', 'network_nodes.csv'), index_col='NODE_ID')

        # AC edges
        self.df_e = pd.read_csv(os.path.join(self.data_dir, 'egrimod-nem-dataset-v1.3', 'akxen-egrimod-nem-dataset-4806603', 'network', 'network_edges.csv'), index_col='LINE_ID')

        # HVDC links
        self.df_hvdc_links = pd.read_csv(os.path.join(self.data_dir, 'egrimod-nem-dataset-v1.3', 'akxen-egrimod-nem-dataset-4806603', 'network', 'network_hvdc_links.csv'), index_col='HVDC_LINK_ID')

        # AC interconnector links
        self.df_ac_i_links = pd.read_csv(os.path.join(self.data_dir, 'egrimod-nem-dataset-v1.3', 'akxen-egrimod-nem-dataset-4806603', 'network', 'network_ac_interconnector_links.csv'), index_col='INTERCONNECTOR_ID')

        # AC interconnector flow limits
        self.df_ac_i_limits = pd.read_csv(os.path.join(self.data_dir, 'egrimod-nem-dataset-v1.3', 'akxen-egrimod-nem-dataset-4806603', 'network', 'network_ac_interconnector_flow_limits.csv'), index_col='INTERCONNECTOR_ID')


        # Generators
        # ----------       
        # Generating unit information
        self.df_g = pd.read_csv(os.path.join(self.data_dir, 'egrimod-nem-dataset-v1.3', 'akxen-egrimod-nem-dataset-4806603', 'generators', 'generators.csv'), index_col='DUID', dtype={'NODE': int})
        
        # Perturb short-run marginal costs (SRMCs) so all unique 
        # (add uniformly distributed random number between 0 and 2 to each SRMC. Set seed so this randomness
        # can be reproduced)
        np.random.seed(seed)
        self.df_g['SRMC_2016-17'] = self.df_g['SRMC_2016-17'] + np.random.uniform(0, 2, self.df_g.shape[0])
        
        
        # Load scenario data
        # ------------------
        with open(os.path.join(scenarios_dir, 'weekly_scenarios.pickle'), 'rb') as f:
            self.df_scenarios = pickle.load(f)
        

class OrganiseData(RawData):
    "Organise data to be used in mathematical program"
    
    def __init__(self, data_dir, scenarios_dir):
        # Load model data
        super().__init__(data_dir, scenarios_dir)
        
        
    def get_admittance_matrix(self):
        "Construct admittance matrix for network"

        # Initialise dataframe
        df_Y = pd.DataFrame(data=0j, index=self.df_n.index, columns=self.df_n.index)

        # Off-diagonal elements
        for index, row in self.df_e.iterrows():
            fn, tn = row['FROM_NODE'], row['TO_NODE']
            df_Y.loc[fn, tn] += - (1 / (row['R_PU'] + 1j * row['X_PU'])) * row['NUM_LINES']
            df_Y.loc[tn, fn] += - (1 / (row['R_PU'] + 1j * row['X_PU'])) * row['NUM_LINES']

        # Diagonal elements
        for i in self.df_n.index:
            df_Y.loc[i, i] = - df_Y.loc[i, :].sum()

        # Add shunt susceptance to diagonal elements
        for index, row in self.df_e.iterrows():
            fn, tn = row['FROM_NODE'], row['TO_NODE']
            df_Y.loc[fn, fn] += (row['B_PU'] / 2) * row['NUM_LINES']
            df_Y.loc[tn, tn] += (row['B_PU'] / 2) * row['NUM_LINES']

        return df_Y
    
    
    def get_HVDC_incidence_matrix(self):
        "Incidence matrix for HVDC links"
        
        # Incidence matrix for HVDC links
        df = pd.DataFrame(index=self.df_n.index, columns=self.df_hvdc_links.index, data=0)

        for index, row in self.df_hvdc_links.iterrows():
            # From nodes assigned a value of 1
            df.loc[row['FROM_NODE'], index] = 1

            # To nodes assigned a value of -1
            df.loc[row['TO_NODE'], index] = -1
        
        return df
    
    
    def get_all_ac_edges(self):
        "Tuples defining from and to nodes for all AC edges (forward and reverse)"
        
        # Set of all AC edges
        edge_set = set()
        
        # Loop through edges, add forward and reverse direction indice tuples to set
        for index, row in self.df_e.iterrows():
            edge_set.add((row['FROM_NODE'], row['TO_NODE']))
            edge_set.add((row['TO_NODE'], row['FROM_NODE']))
        
        return edge_set
    
    
    def get_network_graph(self):
        "Graph containing connections between all network nodes"
        network_graph = {n: set() for n in self.df_n.index}

        for index, row in self.df_e.iterrows():
            network_graph[row['FROM_NODE']].add(row['TO_NODE'])
            network_graph[row['TO_NODE']].add(row['FROM_NODE'])
        
        return network_graph
    
    
    def get_all_dispatchable_fossil_generator_duids(self):
        "Fossil dispatch generator DUIDs"
        
        # Filter - keeping only fossil and scheduled generators
        mask = (self.df_g['FUEL_CAT'] == 'Fossil') & (self.df_g['SCHEDULE_TYPE'] == 'SCHEDULED')
        
        return self.df_g[mask].index    
       
    
    def get_reference_nodes(self):
        "Get reference node IDs"
        
        # Filter Regional Reference Nodes (RRNs) in Tasmania and Victoria.
        mask = (self.df_n['RRN'] == 1) & (self.df_n['NEM_REGION'].isin(['TAS1', 'VIC1']))
        reference_node_ids = self.df_n[mask].index
        
        return reference_node_ids
    
       
    def get_generator_node_map(self, generators):
        "Get set of generators connected to each node"
        
        generator_node_map = (self.df_g.reindex(index=generators)
                              .reset_index()
                              .rename(columns={'OMEGA_G': 'DUID'})
                              .groupby('NODE').agg(lambda x: set(x))['DUID']
                              .reindex(self.df_n.index, fill_value=set()))
        
        return generator_node_map
    
    
    def get_ac_interconnector_summary(self):
        "Summarise aggregate flow limit information for AC interconnectors"

        # Create dicitionary containing collections of AC branches for which interconnectors are defined. Create
        # These collections for both forward and reverse directions.
        interconnector_limits = {}

        for index, row in self.df_ac_i_limits.iterrows():
            # Forward limit
            interconnector_limits[index+'-FORWARD'] = {'FROM_REGION': row['FROM_REGION'], 'TO_REGION': row['TO_REGION'], 'LIMIT': row['FORWARD_LIMIT_MW']}

            # Reverse limit
            interconnector_limits[index+'-REVERSE'] = {'FROM_REGION': row['TO_REGION'], 'TO_REGION': row['FROM_REGION'], 'LIMIT': row['REVERSE_LIMIT_MW']}

        # Convert to DataFrame
        df_interconnector_limits = pd.DataFrame(interconnector_limits).T

        # Find all branches that consitute each interconnector - order is important. 
        # First element is 'from' node, second is 'to node
        branch_collections = {b: {'branches': list()} for b in df_interconnector_limits.index}

        for index, row in self.df_ac_i_links.iterrows():
            # For a given branch, find the interconnector index to which it belongs. This will either be the forward or
            # reverse direction as defined in the interconnector links DataFrame. If the forward direction, 'FROM_REGION'
            # will match between DataFrames, else it indicates the link is in the reverse direction.

            # Assign branch to forward interconnector limit ID
            mask_forward = (df_interconnector_limits.index.str.contains(index) 
                      & (df_interconnector_limits['FROM_REGION'] == row['FROM_REGION']) 
                      & (df_interconnector_limits['TO_REGION'] == row['TO_REGION']))

            # Interconnector ID corresponding to branch 
            branch_index_forward = df_interconnector_limits.loc[mask_forward].index[0]

            # Add branch tuple to branch collection
            branch_collections[branch_index_forward]['branches'].append((row['FROM_NODE'], row['TO_NODE']))

            # Assign branch to reverse interconnector limit ID
            mask_reverse = (df_interconnector_limits.index.str.contains(index) 
                            & (df_interconnector_limits['FROM_REGION'] == row['TO_REGION']) 
                            & (df_interconnector_limits['TO_REGION'] == row['FROM_REGION']))

            # Interconnector ID corresponding to branch 
            branch_index_reverse = df_interconnector_limits.loc[mask_reverse].index[0]

            # Add branch tuple to branch collection
            branch_collections[branch_index_reverse]['branches'].append((row['TO_NODE'], row['FROM_NODE']))

        # Append branch collections to interconnector limits DataFrame
        df_interconnector_limits['branches'] = pd.DataFrame(branch_collections).T['branches']
        
        return df_interconnector_limits

    
class DCOPFModel(OrganiseData):
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
        # Week index
        self.week_index = None
        
        # Scenario index
        self.scenario_index = None
                
        
    def create_model(self):
        "Create model object"
        
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
        network_graph = self.get_network_graph()
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

        def AC_POWER_FLOW_LIMIT_RULE(model, j):
            return model.AC_FLOW[j] - model.F[j] <= 0
        model.AC_POWER_FLOW_LIMIT = Constraint(model.OMEGA_J, rule=AC_POWER_FLOW_LIMIT_RULE)

        
        # Expressions
        # -----------
        # Effective emissions intensity (original emissions intensity x scaling factor)
        def E_HAT_RULE(model, g):
            return model.E[g] * model.EMISSIONS_INTENSITY_SHOCK_FACTOR[g]
        model.E_HAT = Expression(model.OMEGA_G, rule=E_HAT_RULE)
        
        # Total emissions [tCO2]
        model.TOTAL_EMISSIONS = Expression(expr=sum(model.p[g] * model.BASE_POWER * model.E_HAT[g] * model.SCENARIO_DURATION for g in model.OMEGA_G))
        
        # Total energy demand [MWh]
        model.TOTAL_ENERGY_DEMAND = Expression(expr=sum(model.D[n] * model.BASE_POWER * model.SCENARIO_DURATION for n in model.OMEGA_N))

        # Total energy output from regulated generators [MWh]
        model.TOTAL_REGULATED_GENERATOR_ENERGY = Expression(expr=sum(model.p[g] * model.BASE_POWER * model.SCENARIO_DURATION for g in model.OMEGA_G))
        
        # Total energy from intermittent generators [MWh]
        model.TOTAL_INTERMITTENT_ENERGY = Expression(expr=sum(model.INTERMITTENT[n] * model.BASE_POWER * model.SCENARIO_DURATION for n in model.OMEGA_N))
        
        # Average emissions intensity of regulated generators [tCO2/MWh]
        model.AVERAGE_EMISSIONS_INTENSITY_REGULATED_GENERATORS = Expression(expr=model.TOTAL_EMISSIONS / model.TOTAL_REGULATED_GENERATOR_ENERGY)
        
        # Average emissions intensity of system [tCO2/MWh]
        model.AVERAGE_EMISSIONS_INTENSITY_SYSTEM = Expression(expr=model.TOTAL_EMISSIONS / model.TOTAL_ENERGY_DEMAND)
        
        # Net scheme revenue from regulated 'fossil' generators[$]
        model.NET_SCHEME_REVENUE_REGULATED_GENERATORS = Expression(expr=sum((model.E_HAT[g] - model.PHI) * model.p[g] * model.BASE_POWER * model.SCENARIO_DURATION * model.TAU * model.BASE_POWER for g in model.OMEGA_G))
        
        # Net scheme revenue that would need to be paid to intermittent renewables if included in scheme [$]
        model.NET_SCHEME_REVENUE_INTERMITTENT_GENERATORS = Expression(expr= - model.PHI * model.TOTAL_INTERMITTENT_ENERGY * model.TAU * model.BASE_POWER)
        

        # Objective
        # ---------
        # Cost minimisation
        model.OBJECTIVE = Objective(expr=sum((model.C[g] + ((model.E_HAT[g] - model.PHI) * model.TAU)) * model.p[g] for g in model.OMEGA_G))

        
        return model
    
    
    def update_model_parameters(self, week_index, scenario_index, baseline, permit_price):
        """ Update DCOPF model parameters

        Parameters
        ----------
        model : pyomo object
            DCOPF OPF model

        df_scenarios : pandas DataFrame
            Demand and fixed power injection data for each week and each scenario

        week_index : int
            Index of week for which model should be run

        scenario_index : int
            Index of scenario that describes operating condition for the given week

        baseline: float
            Emissions intensity baseline [tCO2/MWh]

        permit price : float
            Permit price [$/tCO2]

        Returns
        -------
        model : pyomo object
            DCOPF model object with updated parameters.    
        """

        # Update fixed nodal power injections
        for n in self.model.OMEGA_N:
            self.model.D[n] = float(self.df_scenarios.loc[('demand', n), (week_index, scenario_index)] / self.model.BASE_POWER.value)
            self.model.HYDRO[n] = float(self.df_scenarios.loc[('hydro', n), (week_index, scenario_index)] / self.model.BASE_POWER.value)
            self.model.INTERMITTENT[n] = float(self.df_scenarios.loc[('intermittent', n), (week_index, scenario_index)] / self.model.BASE_POWER.value)
            
        # Update emissions intensity baseline
        self.model.PHI = float(baseline)

        # Update permit price
        self.model.TAU = float(permit_price / self.model.BASE_POWER.value)
        
        # Scenario duration
        self.model.SCENARIO_DURATION = float(self.df_scenarios.loc[('hours', 'duration'), (week_index, scenario_index)])
        
        # Update week index
        self.week_index = week_index
        
        # Update scenario index
        self.scenario_index = scenario_index
               
            
    def solve_model(self):
        "Solve model"
        
        self.opt.solve(self.model)


# In[2]:


from collections import OrderedDict

from pyomo.environ import *

class MPCModel:
    "Model Predictive Controller used to update emissions intensity baseline"
    
    def __init__(self, generator_index, forecast_interval):
        # Create model object
        self.model = self.mpc_baseline(generator_index, forecast_interval=forecast_interval)       
                
        # Define solver
        self.opt = SolverFactory('gurobi', solver_io='lp')

        
    def mpc_baseline(self, generator_index, forecast_interval):
        """Compute baseline path using model predictive control

        Parameters
        ----------
        generator_index : list
            DUIDs for each generator regulated by the emissions intensity scheme
            
        forecast_interval : int
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
        model.OMEGA_T = Set(initialize=range(1, forecast_interval+1), ordered=True)


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
        model.INITIAL_EMISSIONS_INTENSITY_BASELINE = Param(initialize=0, mutable=True)
        
        # Initial rolling scheme revenue
        model.INITIAL_ROLLING_SCHEME_REVENUE = Param(initialize=0, mutable=True)

        # Rolling scheme revenue target at end of finite horizon
        model.TARGET_ROLLING_SCHEME_REVENUE = Param(initialize=0, mutable=True)
        
        # Indicator denoting whether energy from intermittent renewables is eligible for scheme payments
        model.INTERMITTENT_GENERATORS_REGULATED_INDICATOR = Param(initialize=0, mutable=True)


        # Variables
        # ---------
        # Emissions intensity baseline
        model.phi = Var(model.OMEGA_T)


        # Constraints
        # -----------
        # Scheme revenue must be at target by end of model horizon
        model.SCHEME_REVENUE = Constraint(expr=model.INITIAL_ROLLING_SCHEME_REVENUE 
                                          + sum((model.EMISSIONS_INTENSITY_FORECAST[g, t] - model.phi[t]) * model.ENERGY_FORECAST[g, t] * model.PERMIT_PRICE for g in model.OMEGA_G for t in model.OMEGA_T) 
                                          + sum(model.INTERMITTENT_GENERATORS_REGULATED_INDICATOR[t] * model.INTERMITTENT_ENERGY_FORECAST[t] * model.PERMIT_PRICE for t in model.OMEGA_T)
                                          == model.TARGET_ROLLING_SCHEME_REVENUE)

        # Baseline must be non-negative
        def BASELINE_NONNEGATIVE_RULE(model, t):
            return model.phi[t] >= 0
        model.BASELINE_NONNEGATIVE = Constraint(model.OMEGA_T, rule=BASELINE_NONNEGATIVE_RULE)


        # Objective function
        # ------------------
        # Minimise changes to baseline over finite time horizon
        model.OBJECTIVE = Objective(expr=(((model.phi[model.OMEGA_T.first()] - model.INITIAL_EMISSIONS_INTENSITY_BASELINE) * (model.phi[model.OMEGA_T.first()] - model.INITIAL_EMISSIONS_INTENSITY_BASELINE))
                                          + sum((model.phi[t] - model.phi[t-1]) * (model.phi[t] - model.phi[t-1]) for t in model.OMEGA_T if t > model.OMEGA_T.first()))
                                   )
        return model

            
    def update_model_parameters(self, forecast_generator_emissions_intensity, forecast_generator_energy, forecast_intermittent_energy, intermittent_generators_regulated, permit_price, initial_emissions_intensity_baseline, initial_rolling_scheme_revenue, target_rolling_scheme_revenue):
        """Update parameters used as inputs for the MPC controller

        Parameters
        ----------
        forecast_generator_emissions_intensity : dict
            Expected generator emissions intensities over forecast interval

        forecast_generator_energy : dict
            Forecast weekly energy output from generators over the forecast interval      
            
        forecast_intermittent_energy : dict
            Forecast weekly intermittent energy output from generators over the forecast interval
            
        intermittent_generators_regulated : dict
            Indicator denoting if energy from renewables receives payments under the scheme

        permit_price : float
            Emissions price [tCO2/MWh]

        initial_emissions_intensity_baseline : float
            Emissions intensity baseline implemented for preceding week [tCO2/MWh]

        initial_rolling_scheme_revenue : float
            Rolling scheme revenue at end of preceding week [$]

        target_rolling_scheme_revenue : float
            Target scheme revenue to be obtained in the future [$]
        """

        # For each time interval in the forecast horizon
        for t in self.model.OMEGA_T:
            
            if intermittent_generators_regulated[t]:
                self.model.INTERMITTENT_GENERATORS_REGULATED_INDICATOR[t] = float(1)
            else:
                self.model.INTERMITTENT_GENERATORS_REGULATED_INDICATOR[t] = float(0)
            
            # For each generator
            for g in self.model.OMEGA_G:
                # Predicted generator emissions intensity for future periods
                self.model.EMISSIONS_INTENSITY_FORECAST[g, t] = float(forecast_generator_emissions_intensity[t][g])
                
                # Predicted weekly energy output
                self.model.ENERGY_FORECAST[g, t] = float(forecast_generator_energy[t][g])
                
                # Predicted weekly energy from intermittent generators
                self.model.INTERMITTENT_ENERGY_FORECAST[t] = float(forecast_intermittent_energy[t])

        # Permit price
        self.model.PERMIT_PRICE = float(permit_price)

        # Emissions intensity baseline from previous period
        self.model.INITIAL_EMISSIONS_INTENSITY_BASELINE = float(initial_emissions_intensity_baseline)

        # Initial rolling scheme revenue
        self.model.INITIAL_ROLLING_SCHEME_REVENUE = float(initial_rolling_scheme_revenue)

        # Rolling scheme revenue target at end of forecast horizon
        self.model.TARGET_ROLLING_SCHEME_REVENUE = float(target_rolling_scheme_revenue)
            

    def solve_model(self):
        "Solve for optimal emissions intensity baseline path"

        self.opt.solve(self.model)


    def get_optimal_baseline_path(self):
        "Get optimal emissions intenstiy baseline path based on MPC controller"

        # Optimal emissions intensity baseline path as determined by MPC controller
        optimal_baseline_path = OrderedDict(self.model.phi.get_values())

        return optimal_baseline_path


    def get_next_baseline(self):
        "Get next baseline to be implemented for the coming week"

        # Optimal path of baselines to be implemented over the finite horizon
        optimal_baseline_path = self.get_optimal_baseline_path()

        # Next 'optimal' emissions intensity baseline to implement for the coming interval
        next_baseline = float(optimal_baseline_path[self.model.OMEGA_T.first()])

        return next_baseline


# In[3]:


import os
import time
import hashlib

import numpy as np


# Paths
# -----
# Directory containing network and generator data
data_dir = os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, os.path.pardir, 'data')

# Path to scenarios directory
scenarios_dir = os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, '1_create_scenarios', 'output')  


def run_model(data_dir, scenarios_dir, shock_option, update_mode, description,
              intermittent_generators_regulated=dict(),
              forecast_generator_emissions_intensity=dict(), 
              forecast_generator_energy=dict(),
              forecast_intermittent_generator_energy=dict(),
              forecast_interval_mpc=None,
              week_of_shock=None, 
              default_baseline=0,
              initial_permit_price=40, 
              initial_rolling_scheme_revenue=0, 
              target_scheme_revenue=dict(), 
              seed=10, 
              model_horizon=52):    
    
    """Run model with given parameters

    Parameters
    ----------
    data_dir : str
        Path to directory containing core data files used in model initialisation

    scenarios_dir : str
        Path to directory containing representative operating scenarios

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
            
    description : str
        Description of model / scenario being tested
    
    intermittent_generators_regulated : dict
        Dictionary containing boolean indicators for each week. If 'True' for a given week, intermittent 
        generators are subject to the emissions policy and receive payments under the scheme.
    
    forecast_generator_emissions_intensity : dict
        Forecast generator emissions intensities for future periods [tCO2/MWh]
        
    forecast_generator_energy : dict
        Forecast generator energy output for future periods [MWh]
        
    forecast_intermittent_generator_energy : dict
        Forecast intermittent generator output for future periods [MWh]
        
    forecast_interval_mpc : int or None
        Forecast interval for MPC controller. Default = None.
    
    week_of_shock : int or None
        Index for week at which either generation or emissions intensities shocks will be implemented / begin.
        Default = 10.

    default_baseline : float
        Initialised emissions intensity baseline value [tCO2/MWh]. Default = 1 tCO2/MWh.

    initial_permit_price : float
        Initialised permit price value [$/tCO2]. Default = 40 $/tCO2.
        
    initial_rolling_scheme_revenue : float
        Initialised rolling scheme revenue value [$]. Default = $0.

    target_scheme_revenue : dict
        Net scheme revenue target defined for each period [$]. Default = $0 for all periods.
        
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
    print(f'Running model {update_mode} with run option {shock_option}')

    # Summary of all parameters defining the model
    parameter_values = (shock_option, update_mode, description, intermittent_generators_regulated,
                        forecast_generator_emissions_intensity, forecast_generator_energy, 
                        forecast_interval_mpc, week_of_shock, default_baseline, initial_permit_price, 
                        initial_rolling_scheme_revenue, target_scheme_revenue, seed, model_horizon)
    
    # Convert parameters to string
    parameter_values_string = ''.join([str(i) for i in parameter_values])

    # Find sha256 of parameter values - used as a unique identifier for the model run
    run_id = hashlib.sha256(parameter_values_string.encode('utf-8')).hexdigest()[:8].upper()

    # Summary of model options, identified by the hash value of these options
    run_summary = {run_id: {'shock_option': shock_option, 
                            'update_mode': update_mode,
                            'description': description,
                            'intermittent_generators_regulated': intermittent_generators_regulated,
                            'forecast_generator_emissions_intensity': forecast_generator_emissions_intensity,
                            'forecast_generator_energy': forecast_generator_energy,
                            'forecast_intermittent_generator_energy': forecast_intermittent_generator_energy,
                            'forecast_interval_mpc': forecast_interval_mpc,
                            'week_of_shock': week_of_shock,
                            'default_baseline': default_baseline, 
                            'initial_permit_price': initial_permit_price,
                            'initial_rolling_scheme_revenue': initial_rolling_scheme_revenue, 
                            'target_scheme_revenue': target_scheme_revenue,                            
                            'seed': seed,
                            'model_horizon': model_horizon}
                  }


    # Check if model parameters valid
    # -------------------------------
    if update_mode not in ['NO_UPDATE', 'REVENUE_REBALANCE_UPDATE', 'MPC_UPDATE']:
        raise Warning(f'Unexpected update_mode encountered: {update_mode}')

    if shock_option not in ['NO_SHOCKS', 'EMISSIONS_INTENSITY_SHOCK']:
        raise Warning(f'Unexpected shock_option encountered: {shock_option}')


    # Create model objects
    # --------------------
    # Instantiate DCOPF model object
    DCOPF = DCOPFModel(data_dir=data_dir, scenarios_dir=scenarios_dir)

    # Instantiate Model Predictive Controller if MPC update specified
    if update_mode == 'MPC_UPDATE':
        MPC = MPCModel(generator_index=DCOPF.model.OMEGA_G, forecast_interval=forecast_interval_mpc)
    
    
    # Result containers
    # -----------------
    # Prices at each node for each scenario
    scenario_nodal_prices = dict()
    
    # Power output for each scenario
    scenario_power_output = dict()

    # Results for each scenario
    scenario_metrics = {'net_scheme_revenue_regulated_generators': dict(),
                        'net_scheme_revenue_intermittent_generators': dict(),
                        'total_emissions_tCO2': dict(),
                        'regulated_generator_energy_MWh': dict(),
                        'total_regulated_generator_energy_MWh': dict(),
                        'total_intermittent_energy_MWh': dict(),
                        'total_demand_MWh': dict(),
                        'energy_revenue': dict(),
                       }
    
    # Aggregated results for each week
    week_metrics = {'baseline': dict(),
                    'net_scheme_revenue_regulated_generators': dict(),
                    'net_scheme_revenue_intermittent_generators': dict(),
                    'net_scheme_revenue': dict(),
                    'rolling_scheme_revenue_interval_start': dict(),
                    'rolling_scheme_revenue_interval_end': dict(),
                    'regulated_generator_emissions_intensities': dict(),
                    'total_emissions_tCO2': dict(),
                    'regulated_generator_energy_MWh': dict(),
                    'total_regulated_generator_energy_MWh': dict(),
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
    np.random.seed(seed)

    # Augment original emissions intensity by random scaling factor between 0.8 and 1
    if shock_option == 'EMISSIONS_INTENSITY_SHOCK':
        df_emissions_intensity_shock_factor = pd.Series(index=DCOPF.model.OMEGA_G, data=np.random.uniform(0.8, 1, len(DCOPF.model.OMEGA_G)))


    # Run scenarios
    # -------------
    # Week indices for which model will be run (can run for less than one year by using model_horizon)
    weeks = DCOPF.df_scenarios.columns.levels[0][:model_horizon]
    
    # For each week
    for week_index in weeks:
        # Start clock to see how long it takes to solve all scenarios for each week
        t0 = time.time()
        
        # Initialise policy parameters if the first week
        if week_index == 1:            
            # Initialise permit price
            permit_price = initial_permit_price
            
            # Initialise rolling scheme revenue (value at the end of previous calibration interval)
            rolling_scheme_revenue_interval_start = initial_rolling_scheme_revenue
        
        
        # Update rolling scheme revenue (amount of money in bank account at start of calibration interval)
        if week_index > 1:
            rolling_scheme_revenue_interval_start = week_metrics['rolling_scheme_revenue_interval_end'][week_index-1]

        # Record amount of money in bank account at start of calibration interval
        week_metrics['rolling_scheme_revenue_interval_start'][week_index] = rolling_scheme_revenue_interval_start
        
            
        # Compute baseline
        # ----------------
        if update_mode == 'NO_UPDATE':
            # No update to baseline (should be 0 tCO2/MWh)
            baseline = default_baseline

        elif update_mode == 'REVENUE_REBALANCE_UPDATE':
            # Forecast regulated generator emissions in next period
            forecast_regulated_generator_total_emissions = sum(forecast_generator_emissions_intensity[week_index][1][g] * forecast_generator_energy[week_index][1][g] for g in DCOPF.model.OMEGA_G)

            # Forecast energy output from 'fossil' generators (definitely under policy's remit)
            forecast_fossil_generator_total_energy = sum(forecast_generator_energy[week_index][1][g] for g in DCOPF.model.OMEGA_G)
            
            # Forecast energy from intermittent reneweable generators (may be under scheme's remit)
            forecast_intermittent_generator_total_energy = forecast_intermittent_generator_energy[week_index][1]
            
            # Forecast regulated generator energy in next period. Value may change depending on whether or not
            # intermittent generators are subject to the scheme's remit.
            if intermittent_generators_regulated and intermittent_generators_regulated[week_index]:
                # Intermittent generators are part of scheme's remit, and receive payments
                forecast_regulated_generator_total_energy = forecast_fossil_generator_total_energy + forecast_intermittent_generator_total_energy
            else:
                # Only output from fossil generators considered
                forecast_regulated_generator_total_energy = forecast_fossil_generator_total_energy
                
            # Forecast regulated generator average emissions intensity
            forecast_regulated_generator_average_emissions_intensity = forecast_regulated_generator_total_emissions / forecast_regulated_generator_total_energy

            # Update baseline seeking to re-balance net scheme revenue every period based on forecast output
            baseline = forecast_regulated_generator_average_emissions_intensity - ((target_scheme_revenue[week_index] - rolling_scheme_revenue_interval_start) / (permit_price * forecast_regulated_generator_total_energy))

            # Set baseline to 0 if updated value less than zero
            if baseline < 0:
                baseline = 0
        
        elif update_mode == 'MPC_UPDATE':
            # Update baseline using a Model Predictive Control paradigm. Goal is to minimise control 
            # action (movement of baseline) while achieving target scheme revenue 6 weeks in the future.
            
            # If first week, set previous baseline to expected emissions intensity for coming week
            if week_index == 1:
                baseline = 1.02
                
            # Have limited information at end of model horizon to make forecast e.g. in the 2nd to last
            # week it's not possible to forecast 6 weeks in the future. Instead use the forecast from the
            # last optimal path calculated.
            if week_index <= weeks[-1] - forecast_interval_mpc + 1:
                # Expected generator energy output in forecast interval
                forecast_energy = forecast_generator_energy[week_index]
                
                # Expected generator emissions intensities in forecast interval
                forecast_emissions_intensity = forecast_generator_emissions_intensity[week_index]
                
                # Expected energy output from renewables over forecast interval
                forecast_intermittent_energy = forecast_intermittent_generator_energy[week_index]
                
                                            
                # Update MPC controller parameters
                MPC.update_model_parameters(forecast_generator_emissions_intensity=forecast_emissions_intensity,
                                            forecast_generator_energy=forecast_energy,
                                            forecast_intermittent_generator_energy=forecast_intermittent_energy,
                                            permit_price=permit_price,
                                            initial_emissions_intensity_baseline=baseline,
                                            initial_rolling_scheme_revenue=rolling_scheme_revenue_interval_start,
                                            target_rolling_scheme_revenue=target_scheme_revenue[week_index])
                                
                # Solve model
                MPC.solve_model()
            
                # Get next baseline to be implemented
                baseline_path = MPC.get_optimal_baseline_path()
            
                # Baseline to be implemented next
                baseline = list(baseline_path.items())[0][1]
            
            else:
                # Find index of baseline from optimal path previously calculated
                baseline_path_index = week_index - (weeks[-1] - forecast_interval_mpc)
                
                # Baseline to be implemented
                baseline = list(baseline_path.items())[baseline_path_index-1][1]
                
        # Record baseline applying for the given week
        week_metrics['baseline'][week_index] = baseline
        
        # Apply emissions intensity shock if shock option specified. Shock occurs once at week_index 
        # given by week_of_shock
        if (shock_option == 'EMISSIONS_INTENSITY_SHOCK') and (week_index == week_of_shock):
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
            scenario_metrics['net_scheme_revenue_regulated_generators'][(week_index, scenario_index)] = DCOPF.model.NET_SCHEME_REVENUE_REGULATED_GENERATORS.expr()
            
            # Net scheme revenue that would need to be paid to intermittent renewables if included in scheme [$]
            scenario_metrics['net_scheme_revenue_intermittent_generators'][(week_index, scenario_index)] = DCOPF.model.NET_SCHEME_REVENUE_INTERMITTENT_GENERATORS.expr()
            
            # Total emissions [tCO2]
            scenario_metrics['total_emissions_tCO2'][(week_index, scenario_index)] = DCOPF.model.TOTAL_EMISSIONS.expr()
            
            # Generator energy output
            scenario_metrics['regulated_generator_energy_MWh'][(week_index, scenario_index)] = {g: DCOPF.model.p[g].value * DCOPF.model.BASE_POWER.value * DCOPF.model.SCENARIO_DURATION.value for g in DCOPF.model.OMEGA_G}
            
            # Total emissions from regulated generators [tCO2]
            scenario_metrics['total_regulated_generator_energy_MWh'][(week_index, scenario_index)] = DCOPF.model.TOTAL_REGULATED_GENERATOR_ENERGY.expr()
            
            # Total energy from intermittent generators [MWh]
            scenario_metrics['total_intermittent_energy_MWh'][(week_index, scenario_index)] = DCOPF.model.TOTAL_INTERMITTENT_ENERGY.expr()
            
            # Total system energy demand [MWh]
            scenario_metrics['total_demand_MWh'][(week_index, scenario_index)] = DCOPF.model.TOTAL_ENERGY_DEMAND.expr()
            
            # Total revenue from energy sales (nodal price x nodal demand x scenario duration)
            scenario_metrics['energy_revenue'][(week_index, scenario_index)] = sum(DCOPF.model.dual[DCOPF.model.POWER_BALANCE[n]] * DCOPF.model.BASE_POWER.value * DCOPF.model.D[n].value * DCOPF.model.BASE_POWER.value * DCOPF.model.SCENARIO_DURATION.value for n in DCOPF.model.OMEGA_N)
            

            
        # Compute aggregate statistics for given week
        # -------------------------------------------
        # Net scheme revenue from regulated 'fossil' generators for each week [$]
        week_metrics['net_scheme_revenue_regulated_generators'][week_index] = sum(scenario_metrics['net_scheme_revenue_regulated_generators'][(week_index, s)] for s in DCOPF.df_scenarios.columns.levels[1])
        
        # Net scheme revenue that would need to be paid to intermittent renewables each week if included in scheme [$]
        week_metrics['net_scheme_revenue_intermittent_generators'][week_index] = sum(scenario_metrics['net_scheme_revenue_intermittent_generators'][(week_index, s)] for s in DCOPF.df_scenarios.columns.levels[1])
        
        # Net scheme revenue depending on inclusion of intermittent renewables in schemes [$]
        if intermittent_generators_regulated and intermittent_generators_regulated[week_index]:
            week_metrics['net_scheme_revenue'][week_index] = week_metrics['net_scheme_revenue_regulated_generators'][week_index] + week_metrics['net_scheme_revenue_intermittent_generators'][week_index]
        else:
            week_metrics['net_scheme_revenue'][week_index] = week_metrics['net_scheme_revenue_regulated_generators'][week_index]
        
        # Emissions intensities for regulated generators for the given week
        week_metrics['regulated_generator_emissions_intensities'][week_index] = {g: DCOPF.model.E_HAT[g].expr() for g in DCOPF.model.OMEGA_G}

        # Total emissions [tCO2]
        week_metrics['total_emissions_tCO2'][week_index] = sum(scenario_metrics['total_emissions_tCO2'][(week_index, s)] for s in DCOPF.df_scenarios.columns.levels[1])
        
        # Weekly energy output for each generator
        week_metrics['regulated_generator_energy_MWh'][week_index] = {g: sum(scenario_metrics['regulated_generator_energy_MWh'][(week_index, s)][g] for s in DCOPF.df_scenarios.columns.levels[1]) for g in DCOPF.model.OMEGA_G}
        
        # Total output from generators subjected to emissions intensity policy [MWh]
        week_metrics['total_regulated_generator_energy_MWh'][week_index] = sum(scenario_metrics['total_regulated_generator_energy_MWh'][(week_index, s)] for s in DCOPF.df_scenarios.columns.levels[1])
        
        # Total energy from intermittent generators [MWh] (these incumbent generators are generally not subjected to policy)
        week_metrics['total_intermittent_energy_MWh'][week_index] = sum(scenario_metrics['total_intermittent_energy_MWh'][(week_index, s)] for s in DCOPF.df_scenarios.columns.levels[1])
        
        # Total energy demand in given week [MWh]
        week_metrics['total_demand_MWh'][week_index] = sum(scenario_metrics['total_demand_MWh'][(week_index, s)] for s in DCOPF.df_scenarios.columns.levels[1])
        
        # Average emissions intensity of regulated generators [tCO2/MWh]
        week_metrics['average_emissions_intensity_regulated_generators'][week_index] = week_metrics['total_emissions_tCO2'][week_index] / week_metrics['total_regulated_generator_energy_MWh'][week_index]
        
        # Average emissions intensity of whole system (including renewables) [tCO2/MWh]
        week_metrics['average_emissions_intensity_system'][week_index] = week_metrics['total_emissions_tCO2'][week_index] / week_metrics['total_demand_MWh'][week_index]            
                   
        # Record rolling scheme revenue at end of calibration interval [$]
        week_metrics['rolling_scheme_revenue_interval_end'][week_index] = rolling_scheme_revenue_interval_start + week_metrics['net_scheme_revenue'][week_index]
        
        # Total revenue from energy sales for given week [$]
        week_metrics['energy_revenue'][week_index] = sum(scenario_metrics['energy_revenue'][(week_index, s)] for s in DCOPF.df_scenarios.columns.levels[1])

        # Average energy price [$/MWh]
        week_metrics['average_energy_price'][week_index] = week_metrics['energy_revenue'][week_index] / week_metrics['total_demand_MWh'][week_index]
        
        
        print(f'Completed week {week_index} in {time.time()-t0:.2f}s')

        
    # Save results
    # ------------
    with open(f'output/{run_id}_scenario_nodal_prices.pickle', 'wb') as f:
        pickle.dump(scenario_nodal_prices, f)

    with open(f'output/{run_id}_scenario_power_output.pickle', 'wb') as f:
        pickle.dump(scenario_power_output, f)

    with open(f'output/{run_id}_scenario_metrics.pickle', 'wb') as f:
        pickle.dump(scenario_metrics, f)
        
    with open(f'output/{run_id}_week_metrics.pickle', 'wb') as f:
        pickle.dump(week_metrics, f)
        
    with open(f'output/{run_id}_run_summary.pickle', 'wb') as f:
        pickle.dump(run_summary, f)

    with open(f'output/{run_id}_generators.pickle', 'wb') as f:
        pickle.dump(DCOPF.df_g, f)
        
    if shock_option == 'EMISSIONS_INTENSITY_SHOCK':
        with open(f'output/{run_id}_emissions_intensity_shock_factor.pickle', 'wb') as f:
            pickle.dump(df_emissions_intensity_shock_factor, f)
        
    return run_id


# Run model with different scenarios

# In[4]:


# Parameters
# ----------
model_horizon = 10
week_of_shock = 3

# Business-as-usual case
run_model(data_dir=data_dir,
          scenarios_dir=scenarios_dir,
          shock_option='NO_SHOCKS',
          update_mode='NO_UPDATE',
          description='business as usual case',
          forecast_generator_emissions_intensity=dict(),
          forecast_generator_energy=dict(),
          forecast_interval_mpc=None,
          week_of_shock=None,
          default_baseline=0,
          initial_permit_price=0,
          initial_rolling_scheme_revenue=0,
          target_scheme_revenue=dict(),
          seed=10,
          model_horizon=model_horizon)

# Carbon tax - no shocks
run_model(data_dir=data_dir,
          scenarios_dir=scenarios_dir,
          shock_option='NO_SHOCKS',
          update_mode='NO_UPDATE',
          description='carbon tax - no shocks',
          forecast_generator_emissions_intensity=dict(),
          forecast_generator_energy=dict(),
          forecast_interval_mpc=None,
          week_of_shock=None,
          default_baseline=0,
          initial_permit_price=40,
          initial_rolling_scheme_revenue=0,
          target_scheme_revenue=dict(),
          seed=10,
          model_horizon=model_horizon)

# Carbon tax - emissions intensity shock
run_model(data_dir=data_dir,
          scenarios_dir=scenarios_dir,
          shock_option='EMISSIONS_INTENSITY_SHOCK',
          update_mode='NO_UPDATE',
          description='carbon tax - emissions intensity shock',
          forecast_generator_emissions_intensity=dict(),
          forecast_generator_energy=dict(),
          forecast_interval_mpc=None,
          week_of_shock=week_of_shock,
          default_baseline=0,
          initial_permit_price=40,
          initial_rolling_scheme_revenue=0,
          target_scheme_revenue=dict(),
          seed=10,
          model_horizon=model_horizon)


# In[5]:


# intermittent_generators_regulated
a = {i: False for i in range(1, model_horizon+1)}

{i: {j+1: a[i+j] for j in range(0, forecast_interval)} for i in range(1, model_horizon+1)}



# In[ ]:





# In[6]:


def get_all_run_summaries(results_dir):
    """Collate information summarising the parameters used in each run
    
    Parameters
    ----------
    results_dir : str
        Directory containing model output
    
    
    Returns
    -------
    run_summaries : dict
        Dictionary summarising model parameterisations    
    """
    
    # Find all results summary files
    run_summary_files = [i for i in os.listdir(results_dir) if 'run_summary' in i]

    # Container for dictionaries summarising model runs
    run_summaries = dict()

    # Open each run summary file and compile in a single dictionary
    for i in run_summary_files:
        with open(os.path.join(results_dir, i), 'rb') as f:
            # Load run summary from file
            run_summary = pickle.load(f)
            
            # Append to dictionary collating all run summaries
            run_summaries = {**run_summaries, **run_summary}
            
    return run_summaries

# Directory containing model output
results_dir = os.path.join(os.path.curdir, 'output')

# Summary of parameters used for each run
run_summaries = pd.DataFrame.from_dict(get_all_run_summaries(results_dir), orient='index')
run_summaries


# In[ ]:


# Forecast signals
# ---------------
# 1. For the carbon tax case (no emissions intensity shock)
# - re-balance signal
# - MPC signal

# 2. For the carbon tax + emissions intensity shock case
# - re-balance signal
# - MPC signal

# Construct perfect signals first (test to see if revenue re-balancing is correct)
# Perturb perfect forecast signals


# In[7]:


def get_perfect_forecasts(run_id, forecast_intervals):
    """Get perfect forecast for energy and emissions intensity based on previous model results
    
    Parameters
    ----------
    run_id : str
        ID for run already completed that will result in same emissions / energy output profiles
    
    Returns
    -------
    forecast_generator_energy : dict
        Perfect forecast for generator energy output in each week
    
    forecast_generator_emissions_intensity : dict
        Perfect forecast for generator emissions intensities in each week   
    """
    
    # Load run summary for given run_id
    with open(f'output/{run_id}_run_summary.pickle', 'rb') as f:
        run_summary = pickle.load(f)

    # Max number of intervals for which forecast can be constructed (based on simulation that
    # produces same energy and emissions profile)
    total_intervals = run_summary[f'{run_id}']['model_horizon']
    
    # Load weekly results for given run_id
    with open(f'output/{run_id}_week_metrics.pickle', 'rb') as f:
        week_metrics = pickle.load(f)

    # Forecast energy
    forecast_generator_energy = {i: {j+1: week_metrics['regulated_generator_energy_MWh'][i+j] for j in range(0, forecast_intervals)} for i in range(1, total_intervals+1-forecast_intervals+1)}

    # Forecast generator emissions intensities
    forecast_generator_emissions_intensity = {i: {j+1: week_metrics['regulated_generator_emissions_intensities'][i+j] for j in range(0, forecast_intervals)} for i in range(1, total_intervals+1-forecast_intervals+1)}

    return forecast_generator_energy, forecast_generator_emissions_intensity


# ## Perfect forecasts

# In[8]:


# Revenue re-balance - no shocks
# ------------------------------
# run_id corresponding to previously simulated scenario that doesn't consider updates
forecast_run_id = 'D368B8D5'

# Forecasts to be used in model
forecast_generator_energy, forecast_generator_emissions_intensity = get_perfect_forecasts(run_id=forecast_run_id, forecast_intervals=1)

# Target scheme revenue - revenue neutral policy
target_scheme_revenue = {i: 100e3 * i for i in range(1, run_summaries.loc[forecast_run_id, 'model_horizon']+1)}

# Run model with revenue rebalancing rule
run_id = run_model(data_dir=data_dir,
                   scenarios_dir=scenarios_dir,
                   shock_option='NO_SHOCKS',
                   update_mode='REVENUE_REBALANCE_UPDATE',
                   description='Attempt to re-balance revenue in each period',
                   forecast_generator_emissions_intensity=forecast_generator_emissions_intensity,
                   forecast_generator_energy=forecast_generator_energy,
                   forecast_interval_mpc=None,
                   week_of_shock=None,
                   default_baseline=0,
                   initial_permit_price=40,
                   initial_rolling_scheme_revenue=0,
                   target_scheme_revenue=target_scheme_revenue,
                   seed=10,
                   model_horizon=model_horizon)

# Check that updating rule has worked correctly
with open(f'output/{run_id}_week_metrics.pickle', 'rb') as f:
    week_metrics = pickle.load(f)
pd.DataFrame(week_metrics)


# In[ ]:


# Revenue re-balance - emissions intensity shock
# ----------------------------------------------
# run_id corresponding to previously simulated scenario that doesn't consider updates
forecast_run_id = '1AE01292'

# Forecasts to be used in model
forecast_generator_energy, forecast_generator_emissions_intensity = get_perfect_forecasts(run_id=forecast_run_id, forecast_intervals=1)

# Target scheme revenue - revenue neutral policy
target_scheme_revenue = {i: 0 for i in range(1, run_summaries.loc[forecast_run_id, 'model_horizon']+1)}
target_scheme_revenue = {i: 0 if i < 5 else 500e3*i for i in range(1, run_summaries.loc[forecast_run_id, 'model_horizon']+1)}

# Run model with revenue rebalancing rule
run_id = run_model(data_dir=data_dir,
                   scenarios_dir=scenarios_dir,
                   shock_option='EMISSIONS_INTENSITY_SHOCK',
                   update_mode='REVENUE_REBALANCE_UPDATE',
                   description='Attempt to re-balance revenue in each period',
                   forecast_generator_emissions_intensity=forecast_generator_emissions_intensity,
                   forecast_generator_energy=forecast_generator_energy,
                   forecast_interval_mpc=None,
                   week_of_shock=week_of_shock,
                   default_baseline=0,
                   initial_permit_price=40,
                   initial_rolling_scheme_revenue=0,
                   target_scheme_revenue=target_scheme_revenue,
                   seed=10,
                   model_horizon=model_horizon)

# Check that updating rule has worked correctly
with open(f'output/{run_id}_week_metrics.pickle', 'rb') as f:
    week_metrics = pickle.load(f)
pd.DataFrame(week_metrics)


# In[ ]:


# MPC update - no shocks
# ------------------------------
# run_id corresponding to previously simulated scenario that doesn't consider updates
forecast_run_id = 'D368B8D5'

# Number of forecast intervals to be used when implementing MPC controller
forecast_intervals = 6

# Forecasts to be used in model
forecast_generator_energy, forecast_generator_emissions_intensity = get_perfect_forecasts(run_id=forecast_run_id, forecast_intervals=forecast_intervals)

# Target scheme revenue - revenue neutral policy - defines revenue to be attained at end of forecast interval
target_scheme_revenue = {i: 0 for i in range(1, run_summaries.loc[forecast_run_id, 'model_horizon']+1-forecast_intervals+1)}
target_scheme_revenue[run_summaries.loc[forecast_run_id, 'model_horizon']-forecast_intervals+1] = 1e6

# Run model with revenue rebalancing rule
run_id = run_model(data_dir=data_dir,
                   scenarios_dir=scenarios_dir,
                   shock_option='NO_SHOCKS',
                   update_mode='MPC_UPDATE',
                   description='Use MPC to achieve revenue target at end of horizon',
                   forecast_generator_emissions_intensity=forecast_generator_emissions_intensity,
                   forecast_generator_energy=forecast_generator_energy,
                   forecast_interval_mpc=forecast_intervals,
                   week_of_shock=None,
                   default_baseline=0,
                   initial_permit_price=40,
                   initial_rolling_scheme_revenue=0,
                   target_scheme_revenue=target_scheme_revenue,
                   seed=10,
                   model_horizon=model_horizon)

# Check that updating rule has worked correctly
with open(f'output/{run_id}_week_metrics.pickle', 'rb') as f:
    week_metrics = pickle.load(f)
pd.DataFrame(week_metrics)


# In[ ]:


# MPC update - emissions intensity shock
# --------------------------------------
# run_id corresponding to previously simulated scenario that doesn't consider updates
forecast_run_id = '1AE01292'

# Number of forecast intervals to be used when implementing MPC controller
forecast_intervals = 6

# Forecasts to be used in model
forecast_generator_energy, forecast_generator_emissions_intensity = get_perfect_forecasts(run_id=forecast_run_id, forecast_intervals=forecast_intervals)

# Target scheme revenue - revenue neutral policy
target_scheme_revenue = {i: 0 for i in range(1, run_summaries.loc[forecast_run_id, 'model_horizon']+1)}

# Run model with revenue rebalancing rule
run_id = run_model(data_dir=data_dir,
                   scenarios_dir=scenarios_dir,
                   shock_option='EMISSIONS_INTENSITY_SHOCK',
                   update_mode='MPC_UPDATE',
                   description='Use MPC to achieve revenue target at end of horizon',
                   forecast_generator_emissions_intensity=forecast_generator_emissions_intensity,
                   forecast_generator_energy=forecast_generator_energy,
                   forecast_interval_mpc=forecast_intervals,
                   week_of_shock=week_of_shock,
                   default_baseline=0,
                   initial_permit_price=40,
                   initial_rolling_scheme_revenue=0,
                   target_scheme_revenue=target_scheme_revenue,
                   seed=10,
                   model_horizon=model_horizon)

# Check that updating rule has worked correctly
with open(f'output/{run_id}_week_metrics.pickle', 'rb') as f:
    week_metrics = pickle.load(f)
pd.DataFrame(week_metrics)


# ## Imperfect forecast

# In[ ]:


import matplotlib.pyplot as plt

plt.clf()
pd.DataFrame(week_metrics)['baseline'].plot(drawstyle='steps-post')
plt.show()

