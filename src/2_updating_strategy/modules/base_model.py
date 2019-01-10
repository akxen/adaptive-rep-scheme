
# coding: utf-8

# In[1]:


import os
import pickle
from math import pi

import numpy as np
np.random.seed(10)

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
        
        # Perturb short-run marginal costs (SRMCs) so all unique 
        # (add uniformly distributed random number between 0 and 2 to each SRMC)
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

        # Fixed power injections
        model.R = Param(model.OMEGA_N, initialize=0, mutable=True)
        
        # Maximum power output
        def REGISTERED_CAPACITY_RULE(model, g):
            registered_capacity = float(self.df_g.loc[g, 'REG_CAP'])
            return registered_capacity / model.BASE_POWER
        model.REGISTERED_CAPACITY = Param(model.OMEGA_G, rule=REGISTERED_CAPACITY_RULE)

        # Generation shock indicator parameter (=0 if generation shock specified, 
        # else equals 1 if normal operation). Initialize value to 1 (normal operation)
        model.GENERATION_SHOCK_INDICATOR = Param(model.OMEGA_G, initialize=1, mutable=True)
        
        # Emissions intensity shock indicator parameter. Used to scale original emissions intensities.
        model.EMISSIONS_INTENSITY_SHOCK_FACTOR = Param(model.OMEGA_G, initialize=1, mutable=True)
        
        
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
            return model.p[g] <= model.REGISTERED_CAPACITY[g] * model.GENERATION_SHOCK_INDICATOR[g]
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
        

        # Objective
        # ---------
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
            self.model.R[n] = float((self.df_scenarios.loc[('hydro', n), (week_index, scenario_index)] + self.df_scenarios.loc[('intermittent', n), (week_index, scenario_index)]) / self.model.BASE_POWER.value)
            
        # Update emissions intensity baseline
        self.model.PHI = float(baseline)

        # Update permit price
        self.model.TAU = float(permit_price / self.model.BASE_POWER.value)
        
        # Update week index
        self.week_index = week_index
        
        # Update scenario index
        self.scenario_index = scenario_index
               
            
    def solve_model(self):
        "Solve model"
        
        self.opt.solve(self.model)


# In[2]:


def get_aggregate_weekly_statistics(dcopf_object, scenario_power_output, week_index, baseline):
    """Compute summarised weekly statistics
    
    Parameters
    ----------
    dcopf_object : class
        Contains DCOPF scenario results and underlying data
        
    scenario_power_output : dict
        power output for each generator for each week and each scenario
        
    week_index : int
        Week for which aggregate statistics are to be calculated
        
    baseline : int
        Emissions intensity baseline that applied for the given week
        
        
    Returns
    -------
    output : dict
        Dictionary summarising average emissions intensity of regulated generators, total emissions,
        energy output from regulated generators, and net scheme revenue for given week
        (dict keys: 'average_regulated_emissions_intensity', 'emissions_tCO2', 'energy_MWh', 'net_revenue')
    """
    
    # Power output for a given week
    df = pd.DataFrame(scenario_power_output).loc[:, (week_index, slice(None))].reset_index().melt(id_vars=['index']).rename(columns={'variable_0': 'week', 'variable_1': 'scenario', 'value': 'power_pu'}).astype({'week': int, 'scenario': int})

    # Scenario durations for a given week
    duration = dcopf_object.df_scenarios.loc[('hours', 'duration'), (week_index, slice(None))].reset_index()
    duration.columns = duration.columns.droplevel(level=1)

    # Merge scenario durations
    df = pd.merge(df, duration, how='left', left_on=['week', 'scenario'], right_on=['week', 'scenario'])

    # Merge emissions intensity for each DUID
    df = pd.merge(df, dcopf_object.df_g[['EMISSIONS']], how='left', left_on='index', right_index=True)

    # Compute total energy output from each generator for each scenario [MWh]
    df['energy_MWh'] = df['power_pu'].mul(100).mul(df['hours'])

    # Compute total emissions for each scenario [tCO2]
    df['emissions_tCO2'] = df['energy_MWh'].mul(df['EMISSIONS'])

    # Compute net scheme revenue for each generator for each scenario [$]
    df['net_revenue'] = df['EMISSIONS'].subtract(baseline).mul(permit_price).mul(df['energy_MWh'])

    # Compute total emissions and total energy output
    output = df[['emissions_tCO2', 'energy_MWh', 'net_revenue']].sum().to_dict()

    # Average emissions intensity of generators under emissions policy for given week
    output['average_regulated_emissions_intensity'] = output['emissions_tCO2'] / output['energy_MWh']

    return output


# In[3]:


import time
import hashlib

import numpy as np
np.random.seed(10)



# Paths
# -----
# Directory containing network and generator data
data_dir = os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, os.path.pardir, 'data')

# Path to scenarios directory
scenarios_dir = os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, '1_create_scenarios', 'output')


# def update_model(data_dir, scenarios_dir, update_mode, shock_option, permit_price, baseline, week_of_shock, rolling_scheme_revenue, target_scheme_revenue, update_gain)
"""Run model with given parameters

Parameters
----------
data_dir : str
    Path to directory containing core data files used in model initialisation
    
scenarios_dir : str
    Path to directory containing representative operating scenarios

update_mode : str
    Specifies how baseline should be updated each week. 
    
    Options:
        NO_UPDATE         - Same baseline in next iteration
        HISTORIC_UPDATE   - Recalibrate baseline assuming next week will be the same as the week just past
        HOOKE_UPDATE      - Adjustment is proportional to difference between rolling scheme revenue 
                            and revenue target
    
shock_option : str
    Specifies type of shock to which system will be subjected. Options
    
    Options:
        NO_SHOCKS                 - No shocks to system
        GENERATION_SHOCK          - 5% chance that each generator will be unavailable each week 
                                    beginning from 'week_of_shock'
        EMISSIONS_INTENSITY_SHOCK - Emissions intensity scaled by a random number between 0.8 
                                    and 1 at 'week_of_shock'
        
initial_permit_price : float
    Initialised permit price value [$/tCO2]

initial_baseline : float
    Initialised emissions intensity baseline value [tCO2/MWh]

week_of_shock : int
    Index for week at which either generation or emissions intensities shocks will be implemented / begin.

initial_rolling_scheme_revenue : float
    Initialised rolling scheme revenue value [$]
    
target_scheme_revenue : float
    Net scheme revenue target [$]

update_gain : float
    Gain used to modify magnitude of restoring force when there is a difference rolling 
    scheme revenue and the scheme revenue target
"""

# Model Parameters
# ----------------
for update_mode in ['NO_UPDATE', 'HISTORIC_UPDATE', 'HOOKE_UPDATE']:
    for shock_option in ['NO_SHOCKS', 'GENERATION_SHOCK', 'EMISSIONS_INTENSITY_SHOCK']:
        print(f'Running model {update_mode} with run option {shock_option}')
        
        # Type of scenario to be investigated
        update_mode = update_mode

        # Option for given scenario
        shock_option = shock_option

        # Permit price [$/tCO2]
        permit_price = 40

        # Initial value for emissions intensity baseline [tCO2/MWh]
        baseline = 1

        # Index of week for which the shock (generation / emissions intensity) is applied
        week_of_shock = 2

        # Rolling scheme revenue [$]. Initiliase to 0.
        rolling_scheme_revenue = 0

        # Revenue target
        target_scheme_revenue = 0

        # Gain applied to emissions intensity update
        update_gain = 1


        # Summary of all parameters defining the model
        parameter_values = (update_mode, shock_option, permit_price, baseline, week_of_shock, rolling_scheme_revenue, update_gain)

        # Convert parameters to string
        parameter_values_string = ''.join([str(i) for i in parameter_values])

        # Find sha256 of parameter values - used as a unique identifier for the model run
        run_id = hashlib.sha256(parameter_values_string.encode('utf-8')).hexdigest()[:8].upper()

        # Summary of model options, identified by the hash value of these options
        run_summary = {run_id: {'update_mode': update_mode, 'shock_option': shock_option, 'permit_price': permit_price, 
                                'baseline': baseline, 'week_of_shock': week_of_shock, 
                                'rolling_scheme_revenue': rolling_scheme_revenue,
                                'target_scheme_revenue': target_scheme_revenue, 'update_gain': update_gain}}


        # Check if model parameters valid
        # -------------------------------
        if update_mode not in ['NO_UPDATE', 'HISTORIC_UPDATE', 'HOOKE_UPDATE']:
            raise Warning(f'Unexpected update_mode encountered: {update_mode}')

        if shock_option not in ['NO_SHOCKS', 'GENERATION_SHOCK', 'EMISSIONS_INTENSITY_SHOCK']:
            raise Warning(f'Unexpected shock_option encountered: {shock_option}')


        # Create model object
        # -------------------
        # Instantiate DCOPF model object
        DCOPF = DCOPFModel(data_dir=data_dir, scenarios_dir=scenarios_dir)


        # Result containers
        # -----------------
        # Power output for each generator for each scenario
        scenario_power_output = dict()

        # Prices at each node for each scenario
        scenario_nodal_prices = dict()

        # Emissions intensity baseline
        week_baseline = dict()

        # Rolling scheme revenue
        week_rolling_scheme_revenue = dict()
        
        
        # Random shocks (set seed so these will be the same for each scenario)
        # --------------------------------------------------------------------
        np.random.seed(10)
        # Specify if generator is on / off for a given week (1=generator is available, 0=generator unavailable)
        df_generation_shock_indicator = pd.DataFrame(index=DCOPF.model.OMEGA_G, columns=DCOPF.df_scenarios.columns.levels[0], data=1)

        # Pick a uniformly distributed random number between 0 and 1. If > 0.95 (i.e. a 5% chance)
        # turn generator off for the given week. Only apply generation shocks to weeks after and 
        # including week_of_shock.
        df_generation_shock_indicator.loc[:, week_of_shock:] = df_generation_shock_indicator.loc[:, week_of_shock:].applymap(lambda x: 0 if np.random.uniform(0, 1) > 0.95 else 1)
        
        # Augment original emissions intensity by random scaling factor between 0.8 and 1
        df_emissions_intensity_shock_factor = pd.Series(index=DCOPF.model.OMEGA_G, data=np.random.uniform(0.8, 1, len(DCOPF.model.OMEGA_G)))

        
        # Run scenarios
        # -------------
        # For each week
        for week_index in DCOPF.df_scenarios.columns.levels[0][:5]:
            # Start clock to see how long it takes to solve all scenarios for each week
            t0 = time.time()

            # Record baseline applying for the given week
            week_baseline[week_index] = baseline

            # Apply generation shock if run mode specified, and week index is greater than week_of_shock.
            # Note: Different shock occurs each week after week_of_shock.
            if shock_option == 'GENERATION_SHOCK':
                # Loop through all generators
                for g in DCOPF.model.OMEGA_G:
                    # Initialise all generators to have a state of normal operation for each week
                    DCOPF.model.GENERATION_SHOCK_INDICATOR[g] = float(df_generation_shock_indicator.loc[g, week_index])
            
            
            # Apply emissions intensity shock if run mode specified. Shock occurs once at week index 
            # given by week_of_shock
            elif (shock_option == 'EMISSIONS_INTENSITY_SHOCK') and (week_index == week_of_shock):
                # Loop through generators
                for g in DCOPF.model.OMEGA_G:
                    # Augement generator emissions intensity factor
                    DCOPF.model.EMISSIONS_INTENSITY_SHOCK_FACTOR[g] = float(df_emissions_intensity_shock_factor.loc[g])
                    
                    
            # For each representative scenario approximating a week's operating state
            for scenario_index in DCOPF.df_scenarios.columns.levels[1]:        
                # Update model parameters
                DCOPF.update_model_parameters(week_index=week_index, scenario_index=scenario_index, baseline=baseline, permit_price=permit_price)

                # Solve model
                DCOPF.solve_model()

                
                # Store results
                # -------------
                # Power output from each generator
                scenario_power_output[(week_index, scenario_index)] = DCOPF.model.p.get_values()

                # Nodal prices
                scenario_nodal_prices[(week_index, scenario_index)] = {n: DCOPF.model.dual[DCOPF.model.POWER_BALANCE[n]] for n in DCOPF.model.OMEGA_N}

                
            # Compute aggregate statistics for week just past
            aggregate_weekly_statistics = get_aggregate_weekly_statistics(DCOPF, scenario_power_output, week_index, baseline)

            # Update rolling scheme revenue
            rolling_scheme_revenue += aggregate_weekly_statistics['net_revenue']

            # Record rolling scheme revenue
            week_rolling_scheme_revenue[week_index] = rolling_scheme_revenue


            # Update baseline
            # ---------------
            if update_mode == 'BAU':
                # No update to baseline (should be 0 tCO2/MWh)
                baseline = baseline

            elif update_mode == 'HISTORIC_UPDATE':
                # Update baseline based on historic average emissions intensity and total energy output
                # (assumes next week will be similar to the week just gone)
                baseline += aggregate_weekly_statistics['average_regulated_emissions_intensity'] - update_gain * ((target_scheme_revenue - rolling_scheme_revenue) / (permit_price * aggregate_weekly_statistics['energy_MWh']))

            elif update_mode == 'BASELINE_ADJUSTMENT':
                # Increment last week's baseline by an amount proportional to the different 
                # between rolling scheme revenue and the target, scaled by energy output in the week just past.
                baseline += baseline - update_gain * ((target_scheme_revenue - rolling_scheme_revenue) / (permit_price * aggregate_weekly_statistics['energy_MWh']))

            print(f'Completed week {week_index} in {time.time()-t0:.2f}s')


        # Save results
        # ------------
        with open(f'output/{run_id}_scenario_nodal_prices.pickle', 'wb') as f:
            pickle.dump(scenario_nodal_prices, f)

        with open(f'output/{run_id}_scenario_power_output.pickle', 'wb') as f:
            pickle.dump(scenario_power_output, f)

        with open(f'output/{run_id}_week_baseline.pickle', 'wb') as f:
            pickle.dump(week_baseline, f)

        with open(f'output/{run_id}_run_summary.pickle', 'wb') as f:
            pickle.dump(run_summary, f)
            
        with open(f'output/{run_id}_generation_shock_indicator.pickle', 'wb') as f:
            pickle.dump(df_generation_shock_indicator, f)

        with open(f'output/{run_id}_emissions_intensity_shock_factor.pickle', 'wb') as f:
            pickle.dump(df_emissions_intensity_shock_factor, f)


# In[4]:


with open(f'output/{run_id}_run_summary.pickle', 'rb') as f:
    runs = pickle.load(f)
    
pd.DataFrame.from_dict(runs, orient='index')

