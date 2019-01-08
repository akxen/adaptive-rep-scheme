
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
    
    def __init__(self, data_dir, scenarios_dir):
        
        # Paths to directories
        # --------------------
        self.data_dir = data_dir
        
        
        # Network data
        # ------------
        # Nodes
        self.df_n = pd.read_csv(os.path.join(self.data_dir, 'network_nodes.csv'), index_col='NODE_ID')

        # AC edges
        self.df_e = pd.read_csv(os.path.join(self.data_dir, 'network_edges.csv'), index_col='LINE_ID')

        # HVDC links
        self.df_hvdc_links = pd.read_csv(os.path.join(self.data_dir, 'network_hvdc_links.csv'), index_col='HVDC_LINK_ID')

        # AC interconnector links
        self.df_ac_i_links = pd.read_csv(os.path.join(self.data_dir, 'network_ac_interconnector_links.csv'), index_col='INTERCONNECTOR_ID')

        # AC interconnector flow limits
        self.df_ac_i_limits = pd.read_csv(os.path.join(self.data_dir, 'network_ac_interconnector_flow_limits.csv'), index_col='INTERCONNECTOR_ID')


        # Generators
        # ----------       
        # Generating unit information
        self.df_g = pd.read_csv(os.path.join(self.data_dir, 'generators.csv'), index_col='DUID', dtype={'NODE': int})
        
        
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
        self.opt = SolverFactory('gurobi', solver_io='lp')
        
        
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

        # Fixed power injections
        model.R = Param(model.OMEGA_N, initialize=0, mutable=True)


        # Variables
        # ---------
        # Generator output
        def P_RULE(model, g):
            registered_capacity = float(self.df_g.loc[g, 'REG_CAP'])
            return (0, registered_capacity / model.BASE_POWER)
        model.p = Var(model.OMEGA_G, bounds=P_RULE)

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


        # Compute absolute flows over HVDC links
        # --------------------------------------
        model.x_1 = Var(model.OMEGA_H, within=NonNegativeReals)
        model.x_2 = Var(model.OMEGA_H, within=NonNegativeReals)

        def ABS_HVDC_FLOW_1_RULE(model, h):
            return model.x_1[h] >= model.p_H[h]
        model.ABS_HVDC_FLOW_1 = Constraint(model.OMEGA_H, rule=ABS_HVDC_FLOW_1_RULE)

        def ABS_HVDC_FLOW_2_RULE(model, h):
            return model.x_2[h] >= - model.p_H[h]
        model.ABS_HVDC_FLOW_2 = Constraint(model.OMEGA_H, rule=ABS_HVDC_FLOW_2_RULE)

        def HVDC_FLOW_COST_RULE(model):
            return float(10 / model.BASE_POWER)
        model.HVDC_FLOW_COST = Param(initialize=HVDC_FLOW_COST_RULE)


        # Objective
        # ---------
        model.OBJECTIVE = Objective(expr=sum((model.C[g] + ((model.E[g] - model.PHI) * model.TAU)) * model.p[g] for g in model.OMEGA_G) + sum((model.x_1[h] + model.x_2[h]) * model.HVDC_FLOW_COST for h in model.OMEGA_H))

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
            self.model.R[n] = float((self.df_scenarios.loc[('hydro', n), (week_index, scenario_index)] + self.df_scenarios.loc[('intermittent', n), (week_index, scenario_index)]) / self.model.BASE_POWER.value)

        # Update emissions intensity baseline
        if baseline is not None:
            self.model.PHI = float(baseline)

        # Update permit price
        if permit_price is not None:
            self.model.TAU = float(permit_price / self.model.BASE_POWER.value)
            
        # Update week index
        self.week_index = week_index
        
        # Update scenario index
        self.scenario_index = scenario_index
               
            
    def solve_model(self):
        "Solve model"
        
        self.opt.solve(self.model)
        
        
    def get_scenario_total_emissions(self):
        """Total emissions [tCO2] for each scenario

        Parameters
        ----------
        week_index : int
            Index of week for which model is run

        scenario_index : int
            Index of scenario for which model is run

        Returns
        -------
        scenario_emissions : dict
            Total emissions for each NEM region and national level for given scenario    
        """

        # Power output and emissions
        df = pd.DataFrame({'p': self.model.p.get_values()}).mul(self.model.BASE_POWER.value).join(self.df_g[['EMISSIONS', 'NEM_REGION', 'FUEL_TYPE']], how='left')

        # Dictionary in which to store aggregate results
        emissions_scenario_results = dict()

        # Compute total emissions in each NEM region per hour
        df_emissions = df.groupby('NEM_REGION').apply(lambda x: x.prod(axis=1).sum())

        # Sum to find national total per hour
        df_emissions.loc['NATIONAL'] = df_emissions.sum()

        # Multiply by scenario duration to find total emissions for scenario
        df_emissions = df_emissions.mul(self.df_scenarios.loc[('hours', 'duration'), (self.week_index, self.scenario_index)])

        # Dictionary containing emissions data
        scenario_emissions = df_emissions.to_dict()

        return scenario_emissions
    
    
    def get_scenario_scheme_revenue(self):
        """Scheme revenue for given scenario
        
        Parameters
        ----------
        baseline : float
            Emissions intensity baseline which applied for the given scenario
        
        week_index : int
            Index of week for which model is run
        
        scenario_index : int
            Index of scenario for which model is run
        
        Returns
        -------
        scenario_scheme_revenue : dict
            Scheme revenue for each NEM region as well as national total for given scenario        
        """

        # Power output and emissions
        df = pd.DataFrame({'p': self.model.p.get_values()}).mul(self.model.BASE_POWER.value).join(self.df_g[['EMISSIONS', 'NEM_REGION', 'FUEL_TYPE']], how='left')

        # Scheme revenue for each NEM region per hour
        df_scheme_revenue = df.groupby('NEM_REGION').apply(lambda x: x['EMISSIONS'].subtract(self.model.PHI.value).mul(x['p']).sum())

        # Scheme revenue for nation per hour
        df_scheme_revenue.loc['NATIONAL'] = df_scheme_revenue.sum()

        # Multiply by scenario duration to get total scheme revenue
        df_scheme_revenue = df_scheme_revenue.mul(self.df_scenarios.loc[('hours', 'duration'), (self.week_index, self.scenario_index)])

        # Convert to dictionary
        scenario_scheme_revenue = df_scheme_revenue.to_dict()
        
        return scenario_scheme_revenue
    
    
    def get_scenario_energy_revenue(self):
        """Revenue obtained from energy sales for given scenario [$]
        
        Parameters
        ----------
        week_index : int
            Index of week for which model is run
        
        scenario_index : int
            Index of scenario for which model is run
        
        Returns
        -------
        scenario_energy_revenue : dict
            Revenue from energy sales for each NEM region as well as national total for given scenario        
        """
        
        # Revenue from energy sales
        # -------------------------
        df = pd.DataFrame.from_dict({n: [self.model.dual[self.model.POWER_BALANCE[n]] * self.model.BASE_POWER.value] for n in self.model.OMEGA_N}, columns=['price'], orient='index')

        # Demand for given scenario
        df_demand = self.df_scenarios.loc[('demand'), (self.week_index, self.scenario_index)]
        df_demand.name = 'demand'

        # Total revenue from electricity sales per hour
        df_energy_revenue = df.join(df_demand).join(self.df_n['NEM_REGION'], how='left').groupby('NEM_REGION').apply(lambda x: x['price'].mul(x['demand']).sum())
        df_energy_revenue.loc['NATIONAL'] = df_energy_revenue.sum()

        # Total revenue from energy sales
        df_energy_revenue = df_energy_revenue.mul(self.df_scenarios.loc[('hours', 'duration'), (self.week_index, self.scenario_index)])

        # Add to energy revenue results dictionary
        scenario_energy_revenue = df_energy_revenue.to_dict()
        
        return scenario_energy_revenue


    def get_scenario_generation_by_fuel_type(self):
        """Generation by fuel type for each NEM region

        Parameters
        ----------
        week_index : int
            Index of week for which model is run

        scenario_index : int
            Index of scenario for which model is run


        Returns
        -------
        scenario_generation_by_fuel_type : dict
            Total energy output [MWh] for each type of generating unit for each NEM region as well as national total
        """

        # Power output
        df = pd.DataFrame({'p': self.model.p.get_values()}).mul(self.model.BASE_POWER.value).join(self.df_g[['EMISSIONS', 'NEM_REGION', 'FUEL_TYPE']], how='left')

        # Energy output by fuel type for each NEM region [MWh]
        df_fuel_type_generation = df.groupby(['NEM_REGION', 'FUEL_TYPE'])['p'].sum().mul(self.df_scenarios.loc[('hours', 'duration'), (self.week_index, self.scenario_index)])

        # National total
        df_nation = df_fuel_type_generation.reset_index().groupby('FUEL_TYPE')['p'].sum().reset_index()
        df_nation['NEM_REGION'] = 'NATIONAL'

        # Combine regional and national values and convert to dictionary
        scenario_generation_by_fuel_type = pd.concat([df_fuel_type_generation.reset_index(), df_nation], sort=False).set_index(['NEM_REGION', 'FUEL_TYPE'])['p'].to_dict()

        return scenario_generation_by_fuel_type


# In[2]:


import time

# Paths
# -----
# Directory containing network and generator data
data_dir = os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, os.path.pardir, 'data')

# Path to scenarios directory
scenarios_dir = os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, '1_create_scenarios', 'output')


# Instantiate DCOPF model object
DCOPF = DCOPFModel(data_dir=data_dir, scenarios_dir=scenarios_dir)


# Result containers
# -----------------
# For total emissions results
scenario_total_emissions = dict()

# For scheme revenue results
scenario_scheme_revenue = dict()

# For energy revenue
scenario_energy_revenue = dict()

# For generation by fuel type
scenario_generation_by_fuel_type = dict()

# Baseline
weekly_baseline = dict()

# Rolling scheme revenue
rolling_scheme_revenue = dict()


# Run scenarios
# -------------
# Initialise rolling scheme revenue
rolling_scheme_revenue = 0

# Initialise emissions intensity baseline
baseline = 0.9

# For each week
for week_index in DCOPF.df_scenarios.columns.levels[0]:
    # Start clock to see how long it takes to solve all scenarios for each week
    t0 = time.time()
    
    weekly_baseline[week_index] = baseline
    
    # For each representative scenario approximating a week's operating state
    for scenario_index in DCOPF.df_scenarios.columns.levels[1]:        
        # Update model parameters
        DCOPF.update_model_parameters(week_index=week_index, scenario_index=scenario_index, baseline=baseline, permit_price=40)

        # Solve model
        DCOPF.solve_model()


        # Extract results from model object
        # ---------------------------------
        # Emissions
        scenario_total_emissions[(week_index, scenario_index)] = DCOPF.get_scenario_total_emissions()

        # Scheme revenue
        scenario_scheme_revenue[(week_index, scenario_index)] = DCOPF.get_scenario_scheme_revenue()

        # Revenue from energy sales
        scenario_energy_revenue[(week_index, scenario_index)] = DCOPF.get_scenario_energy_revenue()

        # Generation by fuel type
        scenario_generation_by_fuel_type[(week_index, scenario_index)] = DCOPF.get_scenario_generation_by_fuel_type()
        
        
        # Record rolling scheme revenue and update baseline
        # -------------------------------------------------
        # Update rolling scheme revenue
        rolling_scheme_revenue += scenario_scheme_revenue[(week_index, scenario_index)]['NATIONAL']
        
        # Update baseline
        
        
    print(f'Completed week {week_index} in {time.time()-t0:2f}s')


# * Total emissions for each NEM region and nation
# * Total scheme revenue for each NEM region and nation
# * Generation by fuel type and NEM region and nation
# 
# Then aggregate scenarios into weekly values
# * Average price for each NEM region
# * Emissions intensity for system and regulated generators, for nation and each NEM region
# * Generation by fuel type and NEM region and nation

# In[8]:


import matplotlib.pyplot as plt


class ProcessScenarioResults(OrganiseData):
    "Aggregate scenario results to weekly statistics"
    
    def __init__(self, data_dir, scenarios_dir):
        super().__init__(data_dir, scenarios_dir)
        
        self.df_weekly_fixed_injections = self.get_weekly_fixed_energy_injections()

        
    def get_weekly_fixed_energy_injections(self):
        "Fixed demand, hydro, and intermittent injections aggregated to weekly level"

        # Fixed injections for NEM regions
        df_weekly_fixed_injections_regional = self.df_scenarios.T.drop(('hours', 'duration'), axis=1, level=0).mul(self.df_scenarios.T.loc[:, ('hours', 'duration')], axis=0).reset_index().drop('scenario', axis=1, level=0).sort_index(axis=1).groupby('week').sum().T.join(self.df_n[['NEM_REGION']], how='left').reset_index().drop('NODE_ID', axis=1).groupby(by=['level', 'NEM_REGION']).sum().reset_index()

        # Sum to get national total fixed energy injections
        df_weekly_fixed_injections_national = df_weekly_fixed_injections_regional.groupby(by=['level']).sum().reset_index()
        df_weekly_fixed_injections_national['NEM_REGION'] = 'NATIONAL'

        # Concatenate so all data is in same DataFrame
        df_weekly_fixed_injections = pd.concat([df_weekly_fixed_injections_regional, df_weekly_fixed_injections_national], sort=False).set_index(['level', 'NEM_REGION']).T

        return df_weekly_fixed_injections
    
    
    def get_weekly_average_energy_price(self, scenario_energy_revenue):
        "Compute average prices"
        
        # Weekly revenue from energy sales for each NEM region and national total
        df_weekly_energy_revenue = pd.DataFrame.from_dict(scenario_energy_revenue, orient='index').reset_index().drop('level_1', axis=1).groupby('level_0').sum()

        # Average weekly wholesale price [$/MWh]
        df_weekly_average_energy_price = df_weekly_energy_revenue.div(self.df_weekly_fixed_injections.loc[:, 'demand'])
        
        return df_weekly_average_energy_price
    
    
    def get_weekly_emissions(self, scenario_total_emissions):
        "Total emissions each week"
        
        # Total weekly emissions for each NEM zone and national total
        df_weekly_emissions = pd.DataFrame.from_dict(scenario_total_emissions, orient='index').reset_index().drop('level_1', axis=1).groupby('level_0').sum()
        
        return df_weekly_emissions

    
    def get_weekly_system_emissions_intensity(self, scenario_total_emissions):
        "Average emissions of system each week"
        
        # Total weekly emissions for each NEM zone and national total
        df_weekly_emissions = self.get_weekly_emissions(scenario_total_emissions)
        
        # Average system emissions intensity for each each week
        df_weekly_system_emissions_intensity = df_weekly_emissions.div(self.df_weekly_fixed_injections.loc[:, 'demand'])

        return df_weekly_system_emissions_intensity

    
    def get_weekly_regulated_generators_emissions_intensity(self, scenario_total_emissions, scenario_generation_by_fuel_type):
        "Average emissions intensity for regulated generators each week"
        
        # Total emissions each week
        df_weekly_emissions = self.get_weekly_emissions(scenario_total_emissions)
        
        # Weekly energy output from generators subject to the emissions reduction scheme
        df_weekly_regulated_generator_energy_output = pd.DataFrame(scenario_generation_by_fuel_type).T.reset_index().drop('level_1', axis=1, level=0).sort_index(axis=1).groupby('level_0').sum().T.reset_index().drop('level_1', axis=1).groupby('level_0').sum().T

        # Average weekly emissions intensity of regulated generators
        df_weekly_regulated_generators_emissions_intensity = df_weekly_emissions.div(df_weekly_regulated_generator_energy_output).fillna(0)

        return df_weekly_regulated_generators_emissions_intensity

    
    def get_weekly_generation_by_fuel_type(self, scenario_generation_by_fuel_type):
        "Aggregate weekly energy output by fuel type"
        
        # Generation by fuel type - regional and national statistics
        df_weekly_generation_by_fuel_type = pd.DataFrame(scenario_generation_by_fuel_type).T.reset_index().sort_index(axis=1).drop('level_1', axis=1).groupby('level_0').sum()
        
        return df_weekly_generation_by_fuel_type
    
    
# Class object used to process scenario results and produce weekly statistics
WeeklyResults = ProcessScenarioResults(data_dir, scenarios_dir)

# Average energy price each week - regional and national statistics
weekly_average_energy_price = WeeklyResults.get_weekly_average_energy_price(scenario_energy_revenue)

# Average system emissions intensity each week - regional and national statistics
weekly_system_emissions_intensity = WeeklyResults.get_weekly_system_emissions_intensity(scenario_total_emissions)

# Average emissions intensity of generators subject to emissions policy - regional and national statistics
weekly_regulated_generators_emissions_intensity = WeeklyResults.get_weekly_regulated_generators_emissions_intensity(scenario_total_emissions, scenario_generation_by_fuel_type)

# Generation by fuel type
weekly_generation_by_fuel_type = WeeklyResults.get_weekly_generation_by_fuel_type(scenario_generation_by_fuel_type)

