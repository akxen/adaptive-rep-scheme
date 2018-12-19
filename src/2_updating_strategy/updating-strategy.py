
# coding: utf-8

# # Refunded Emissions Payment Updating Strategy
# Strategy to update an observable metric, namely the emissions intensity baseline, which is used to augment generator short-run marginal costs.
# 
# ## Import packages

# In[1]:


import os
import re
import time
import pickle
import itertools
from math import pi

import numpy as np
import pandas as pd

from pyomo.environ import *

import matplotlib.pyplot as plt
np.random.seed(10)


# ## Paths

# In[2]:


class DirectoryPaths(object):
    
    def __init__(self):
        self.data_dir = os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, 'data')
        self.output_dir = os.path.join(os.path.curdir, 'output')

paths = DirectoryPaths()


# ## Model data

# In[3]:


class RawData(object):
    
    def __init__(self):
        
        # Paths to directories
        DirectoryPaths.__init__(self)
        
        
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
        self.df_g['SRMC_2016-17'] = self.df_g['SRMC_2016-17'].map(lambda x: x + np.random.uniform(0, 2))
        
        # Station owners
        self.df_g_own = (pd.read_csv(os.path.join(self.data_dir, 'PUBLIC_DVD_STATIONOWNER_201706010000.CSV'), 
                                    skiprows=1, skipfooter=1, engine='python', parse_dates=['LASTCHANGED'])
                         .sort_values('LASTCHANGED', ascending=False)
                         .drop_duplicates('STATIONID', keep='first'))

        
        # Signals
        # -------
        with open(os.path.join(os.path.curdir, os.path.pardir, '1_create_scenarios', 'output', 'weekly_scenarios.pickle'), 'rb') as f:
            self.df_scenarios = pickle.load(f)
        
        # Sort multi-index
        self.df_scenarios.sort_index(ascending=True, inplace=True)
        

# Create object containing raw model data
raw_data = RawData() 


# ## Organise model data

# In[4]:


class OrganiseData(object):
    "Organise data to be used in mathematical program"
    
    def __init__(self):
        # Load model data
        RawData.__init__(self)
        
        def add_participantids_to_generator_dataframe(self):
            "Add station owner IDs to generators DataFrame"
            
            # New generators DataFrame - merging Participant IDs using Station IDs
            df_g = (self.df_g.reset_index()
                    .merge(self.df_g_own[['PARTICIPANTID', 'STATIONID']], left_on='STATIONID', right_on='STATIONID')
                    .set_index('DUID'))
            return df_g
        
        self.df_g = add_participantids_to_generator_dataframe(self)
        
        def get_aggregate_scenario_energy_demand(self):
            "Regional and national aggregate scenario demand"

            # Total NEM region demand for each operating scenario
            scenario_demand = self.df_scenarios.join(pd.concat([self.df_n[['NEM_REGION']]], axis=1, keys=['REGION'])).loc['demand', :].groupby(('REGION', 'NEM_REGION')).sum()

            # National demand
            scenario_demand.loc['NATIONAL'] = scenario_demand.sum()
            
            # Multiply by duration
            scenario_energy_demand = scenario_demand * self.df_scenarios.loc[('hours', 'duration'), :]

            return scenario_energy_demand
        self.df_aggregate_scenario_energy_demand = get_aggregate_scenario_energy_demand(self)
        
        def get_aggregate_scenario_intermittent_energy_output(self):
            "Regional and national aggregate intermittent energy output demand"

            # Total NEM region intermittent power output for each operating scenario
            scenario_intermittent = self.df_scenarios.join(pd.concat([self.df_n[['NEM_REGION']]], axis=1, keys=['REGION'])).loc['intermittent', :].groupby(('REGION', 'NEM_REGION')).sum()

            # National intermittent power output
            scenario_intermittent.loc['NATIONAL'] = scenario_intermittent.sum()
            
            # Multiply by duration to get energy output
            scenario_intermittent_energy = scenario_intermittent * self.df_scenarios.loc[('hours', 'duration'), :]

            return scenario_intermittent_energy
        self.df_aggregate_scenario_intermittent_energy_output = get_aggregate_scenario_intermittent_energy_output(self)      
        
        def get_aggregate_scenario_hydro_energy_output(self):
            "Regional and national aggregate hydro energy output"

            # Total NEM region hydro power output for each operating scenario
            scenario_hydro = self.df_scenarios.join(pd.concat([self.df_n[['NEM_REGION']]], axis=1, keys=['REGION'])).loc['hydro', :].groupby(('REGION', 'NEM_REGION')).sum()

            # National intermittent power output
            scenario_hydro.loc['NATIONAL'] = scenario_hydro.sum()
            
            # Multiply by duration to get energy output
            scenario_hydro_energy = scenario_hydro * self.df_scenarios.loc[('hours', 'duration'), :]

            return scenario_hydro_energy
        self.df_aggregate_scenario_hydro_energy_output = get_aggregate_scenario_hydro_energy_output(self)
        
        
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
        for index, row in model_data.df_e.iterrows():
            edge_set.add((row['FROM_NODE'], row['TO_NODE']))
            edge_set.add((row['TO_NODE'], row['FROM_NODE']))
        
        return edge_set
    
    def get_network_graph(self):
        "Graph containing connections between all network nodes"
        network_graph = {n: set() for n in model_data.df_n.index}

        for index, row in model_data.df_e.iterrows():
            network_graph[row['FROM_NODE']].add(row['TO_NODE'])
            network_graph[row['TO_NODE']].add(row['FROM_NODE'])
        
        return network_graph
    
    
    def get_all_dispatchable_fossil_generator_duids(self):
        "Fossil dispatch generator DUIDs"
        
        # Filter - keeping only fossil and scheduled generators
        mask = (model_data.df_g['FUEL_CAT'] == 'Fossil') & (model_data.df_g['SCHEDULE_TYPE'] == 'SCHEDULED')
        
        return model_data.df_g[mask].index    
    
    
    def get_intermittent_dispatch(self):
        "Dispatch from intermittent generators (solar, wind)"
        
        # Intermittent generator DUIDs
        intermittent_duids_mask = model_data.df_g['FUEL_CAT'].isin(['Wind', 'Solar'])
        intermittent_duids = model_data.df_g.loc[intermittent_duids_mask].index

        # Intermittent dispatch aggregated by node
        intermittent_dispatch =(model_data.df_dispatch.reindex(columns=intermittent_duids, fill_value=0)
                                .T
                                .join(model_data.df_g[['NODE']])
                                .groupby('NODE').sum()
                                .reindex(index=model_data.df_n.index, fill_value=0)
                                .T)
        
        # Make sure columns are of type datetime
        intermittent_dispatch.index = intermittent_dispatch.index.astype('datetime64[ns]')
        
        return intermittent_dispatch
    
    
    def get_hydro_dispatch(self):
        "Dispatch from hydro plant"
        
        # Dispatch from hydro plant
        hydro_duids_mask = self.df_g['FUEL_CAT'].isin(['Hydro'])
        hydro_duids = self.df_g.loc[hydro_duids_mask].index

        # Hydro plant dispatch aggregated by node
        hydro_dispatch = (self.df_dispatch.reindex(columns=hydro_duids, fill_value=0)
                          .T
                          .join(model_data.df_g[['NODE']])
                          .groupby('NODE').sum()
                          .reindex(index=self.df_n.index, fill_value=0)
                          .T)
        
        # Make sure columns are of type datetime
        hydro_dispatch.index = hydro_dispatch.index.astype('datetime64[ns]')
        
        return hydro_dispatch
    
    
    def get_reference_nodes(self):
        "Get reference node IDs"
        
        # Filter Regional Reference Nodes (RRNs) in Tasmania and Victoria.
        mask = (model_data.df_n['RRN'] == 1) & (model_data.df_n['NEM_REGION'].isin(['TAS1', 'VIC1']))
        reference_node_ids = model_data.df_n[mask].index
        
        return reference_node_ids
    
    
    def get_node_demand(self):   
        "Compute demand at each node for a given time period, t"

        def _node_demand(row):
            # NEM region for a given node
            region = row['NEM_REGION']

            # Load at node
            demand = self.df_load.loc[:, region] * row['PROP_REG_D']

            return demand
        node_demand = self.df_n.apply(_node_demand, axis=1).T
        
        return node_demand
    
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

        # Check that from and to regions conform with regional power flow limit directions
        def check_flow_direction(row):
            if (row['FROM_REGION'] == self.df_ac_i_limits.loc[row.name, 'FROM_REGION']) & (row['TO_REGION'] == model_data.df_ac_i_limits.loc[row.name, 'TO_REGION']):
                return True
            else:
                return False
        # Flow directions are consistent between link and limit DataFrames if True
        flow_directions_conform = self.df_ac_i_links.apply(check_flow_direction, axis=1).all()
        if flow_directions_conform:
            print('Flow directions conform with regional flow limit directions: {0}'.format(flow_directions_conform))
        else:
            raise(Exception('Link flow directions inconsitent with regional flow forward limit definition'))

        # Forward limit
        df_forward = self.df_ac_i_links.apply(lambda x: (x['FROM_NODE'], x['TO_NODE']), axis=1).reset_index().groupby('INTERCONNECTOR_ID').agg(lambda x: list(x)).join(model_data.df_ac_i_limits['FORWARD_LIMIT_MW'], how='left').rename(columns={0: 'branches', 'FORWARD_LIMIT_MW': 'limit'})
        df_forward['new_index'] = df_forward.apply(lambda x: x.name + '-FORWARD', axis=1)
        df_forward.set_index('new_index', inplace=True)

        # Reverse limit
        df_reverse = self.df_ac_i_links.apply(lambda x: (x['TO_NODE'], x['FROM_NODE']), axis=1).reset_index().groupby('INTERCONNECTOR_ID').agg(lambda x: list(x)).join(model_data.df_ac_i_limits['REVERSE_LIMIT_MW'], how='left').rename(columns={0: 'branches', 'REVERSE_LIMIT_MW': 'limit'})
        df_reverse['new_index'] = df_reverse.apply(lambda x: x.name + '-REVERSE', axis=1)
        df_reverse.set_index('new_index', inplace=True)
        df_ac_limits = pd.concat([df_forward, df_reverse])

        return df_ac_limits
    
# Create object containing organised model data
model_data = OrganiseData()


# ## Model

# In[5]:


def create_model(use_pu=False):
    "Create model object"
    
    # Initialise model
    model = ConcreteModel()

    # Sets
    # ----   
    # Nodes
    model.OMEGA_N = Set(initialize=model_data.df_n.index)

    # Generators
    model.OMEGA_G = Set(initialize=model_data.get_all_dispatchable_fossil_generator_duids())

    # AC edges
    ac_edges = model_data.get_all_ac_edges()
    model.OMEGA_NM = Set(initialize=ac_edges)
    
    # Sets of branches for which aggregate AC interconnector limits are defined
    ac_limits = model_data.get_ac_interconnector_summary()
    model.OMEGA_J = Set(initialize=ac_limits.index)
    
    # HVDC links
    model.OMEGA_H = Set(initialize=model_data.df_hvdc_links.index)


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
        return float(model_data.df_g.loc[g, 'EMISSIONS'])
    model.E = Param(model.OMEGA_G, rule=E_RULE)
    
    # Admittance matrix
    admittance_matrix = model_data.get_admittance_matrix()
    def B_RULE(model, n, m):
        if use_pu:
            return float(np.imag(admittance_matrix.loc[n, m]))
        else:
            return model.BASE_POWER * float(np.imag(admittance_matrix.loc[n, m]))
    model.B = Param(model.OMEGA_NM, rule=B_RULE)

    # Reference nodes
    reference_nodes = model_data.get_reference_nodes()
    def S_RULE(model, n):
        if n in reference_nodes:
            return 1
        else:
            return 0
    model.S = Param(model.OMEGA_N, rule=S_RULE)

    # Generator short-run marginal costs
    def C_RULE(model, g):
        marginal_cost = float(model_data.df_g.loc[g, 'SRMC_2016-17'])
        if use_pu:
            return marginal_cost / model.BASE_POWER
        else:
            return marginal_cost
    model.C = Param(model.OMEGA_G, rule=C_RULE)

    # Demand
    model.D = Param(model.OMEGA_N, initialize=0, mutable=True)
    
    # Max voltage angle difference between connected nodes
    model.THETA_DELTA = Param(initialize=float(pi / 2))
    
    # HVDC incidence matrix
    hvdc_incidence_matrix = model_data.get_HVDC_incidence_matrix()
    def K_RULE(model, n, h):
        return float(hvdc_incidence_matrix.loc[n, h])
    model.K = Param(model.OMEGA_N, model.OMEGA_H, rule=K_RULE)    
    
    # Aggregate AC interconnector flow limits
    def F_RULE(model, j):
        power_flow_limit = float(ac_limits.loc[j, 'limit'])
        if use_pu:
            return power_flow_limit / model.BASE_POWER
        else:
            return power_flow_limit
    model.F = Param(model.OMEGA_J, rule=F_RULE)
    
    # Fixed power injections
    model.R = Param(model.OMEGA_N, initialize=0, mutable=True)
    
    
    # Variables
    # ---------
    # Generator output
    def P_RULE(model, g):
        registered_capacity = float(model_data.df_g.loc[g, 'REG_CAP'])
        if use_pu:
            return (0, registered_capacity / model.BASE_POWER)
        else:
            return (0, registered_capacity)
    model.p = Var(model.OMEGA_G, bounds=P_RULE)

    # HVDC flows
    def P_H_RULE(model, h):
        forward_flow_limit = float(model_data.df_hvdc_links.loc[h, 'FORWARD_LIMIT_MW'])
        reverse_flow_limit = float(model_data.df_hvdc_links.loc[h, 'REVERSE_LIMIT_MW'])
        if use_pu:
            return (- reverse_flow_limit / model.BASE_POWER, forward_flow_limit / model.BASE_POWER)
        else:
            return (- reverse_flow_limit, forward_flow_limit)
    model.p_H = Var(model.OMEGA_H, bounds=P_H_RULE)
    
    # Node voltage angles
    model.theta = Var(model.OMEGA_N)


    # Constraints
    # -----------
    # Power balance
    generator_node_map = model_data.get_generator_node_map(model.OMEGA_G)
    network_graph = model_data.get_network_graph()
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
    

    # Objective
    # ---------
    model.x_1 = Var(model.OMEGA_J, within=NonNegativeReals)
    model.x_2 = Var(model.OMEGA_J, within=NonNegativeReals)
    
    def ABS_HVDC_FLOW_1_RULE(model, j):
        return model.x_1[j] >= model.AC_FLOW[j]
    model.ABS_HVDC_FLOW_1 = Constraint(model.OMEGA_J, rule=ABS_HVDC_FLOW_1_RULE)
    
    def ABS_HVDC_FLOW_2_RULE(model, j):
        return model.x_2[j] >= - model.AC_FLOW[j]
    model.ABS_HVDC_FLOW_2 = Constraint(model.OMEGA_J, rule=ABS_HVDC_FLOW_2_RULE)
    
    def HVDC_FLOW_COST_RULE(model):
        if use_pu:
            return 10 / model.BASE_POWER
        else:
            return 10
    model.HVDC_FLOW_COST = Param(initialize=HVDC_FLOW_COST_RULE)
    
    model.OBJECTIVE = Objective(expr=sum((model.C[g] + ((model.E[g] - model.PHI) * model.TAU)) * model.p[g] for g in model.OMEGA_G) + sum((model.x_1[j] + model.x_2[j]) * model.HVDC_FLOW_COST for j in model.OMEGA_J))

    return model


# Initialise model object

# In[6]:


model = create_model(use_pu=True)


# Setup solver

# In[7]:


# Setup solver
# ------------
solver = 'gurobi'
solver_io = 'lp'
model.dual = Suffix(direction=Suffix.IMPORT)
opt = SolverFactory(solver, solver_io=solver_io)


# Function used to update parameters and solve model for specific scenario

# In[8]:


def solve_scenario(model, use_pu, permit_price, baseline, week, scenario, tee=False):
    """Update demand and fixed power injection parameters and solve model
    
    Parameters
    ----------
    use_pu : bool
        Indicator if p.u. normalisation is to be employed
        
    permit_price : float
        Magnitude of permit price
    
    baseline : float
        Emissions intensity baseline to be applied to the given scenario
        
    week : int
        Week number
    
    scenario : int
        Operating scenario number associated with given week
        
    tee : bool
        Indicator if solver should stream output
        
    Returns
    -------
    df : pandas DataFrame
        Formatted DataFrame containing results for given interval
    """
    
    # Update permit price
    if use_pu:
        model.TAU = permit_price / model.BASE_POWER.value
    else:
        model.TAU = permit_price
    
    # Update emissions intensity baseline
    model.PHI = baseline
    
    # Update parameters for each node
    for n in model.OMEGA_N:
        # Node demand
        node_demand = model_data.df_scenarios.loc[('demand', n), (week, scenario)]

        # Hydro power injection
        node_hydro = model_data.df_scenarios.loc[('hydro', n), (week, scenario)]

        # Intermittent power injection
        node_intermittent = model_data.df_scenarios.loc[('intermittent', n), (week, scenario)]

        # If using per-unit scaling
        if use_pu:
            model.D[n] = node_demand / model.BASE_POWER.value
            model.R[n] = (node_hydro + node_intermittent) / model.BASE_POWER.value
        else:
            model.D[n] = node_demand
            model.R[n] = node_hydro + node_intermittent
    
    # Solve model
    res = opt.solve(model, keepfiles=False, tee=tee, warmstart=False)
    model.solutions.store_to(res)

    # Place results in DataFrame
    df = pd.DataFrame(res['Solution'][0])
    
    # Add base power value to DataFrame in case variables need to be re-scaled following p.u. normalisation
    df['Parameter'] = np.nan
    df.loc['BASE_POWER', 'Parameter'] = model.BASE_POWER.value
    
    return df


# Functions used to process individual scenario results for a given week.

# In[9]:


def get_average_electricity_price(df, use_pu, week, scenario):
    """Compute average electricity price given a results DataFrame
    
    Param
    -----
    df : pandas DataFrame
        DataFrame containing model results
        
    use_pu : bool
        Indicator if p.u. normalisation employed
        
    week : int
        Week number
        
    scenario : int
        Operating scenario number associated with given week
        
    Returns
    -------
    df_average_prices : pandas DataFrame
        DataFrame containing average regional and national wholesale electricity prices    
    """
    
    # Factor by which to scale prices and power quanties if p.u. normalisation used
    if use_pu:
        normalisation_factor = df.loc['BASE_POWER', 'Parameter']
    else:
        normalisation_factor = 1
    
    # Filter power-balance results
    mask = df.index.str.contains('POWER_BALANCE')
    
    # Filtered DataFrame
    df_f = df.loc[mask].copy()

    # Extract node IDs
    df_f['node'] = df_f.apply(lambda x: int(re.findall(r'\[(\d+)\]', x.name)[0]), axis=1)

    # Dual variable value
    df_f['value'] = df_f.apply(lambda x: x['Constraint']['Dual'], axis=1)

    # Check if price normalisation has been used
    if df_f['value'].max() > 10:
        raise(Warning('Per-unit normalisation probably not be used. May need to re-scale'))

    # Join demand data
    df_f['demand'] = df_f.apply(lambda x: model_data.df_scenarios.loc[('demand', x['node']), (week, scenario)], axis=1)

    # Join NEM regions
    df_f = df_f.merge(model_data.df_n[['NEM_REGION']], left_on='node', right_index=True, how='left')

    # Prices for each NEM region
    df_average_prices = df_f.groupby('NEM_REGION').apply(lambda x: x['value'].mul(x['demand']).mul(normalisation_factor).sum() / x['demand'].sum())

    # Add national average price
    df_average_prices.loc['NATIONAL'] = df_f['value'].mul(df_f['demand']).mul(normalisation_factor).sum() / df_f['demand'].sum()
    
    # Total revenue (national)
    total_revenue = df_f.apply(lambda x: x['value'] * x['demand'] * normalisation_factor, axis=1).sum()
    
    # Total demand (national)
    total_demand = df_f['demand'].sum()

    return {'average_price': df_average_prices}


def get_scenario_revenue_and_emissions(df, use_pu, baseline, permit_price, week, scenario):
    """Get total net REP scheme revenue and emissions for a given operating scenario
    
    Params
    ------
    df : pandas DataFrame
        Results for given scenario
    
    use_pu : bool
        Indicator if p.u. normalisation has been employed
        
    baseline : float
        Emissions intensity baseline for given scenario
    
    permit_price : float
        Permit price for given scenario
    
    week : int
        Week number
    
    scenario : int
        Operating scenario number associated with given week
    
    Returns
    -------
    scenario_revenue_and_emissions : dict
        Scenario revenue and emissions results
    """
    
    
    # Factor by which to scale prices and power quanties if p.u. normalisation used
    if use_pu:
        normalisation_factor = df.loc['BASE_POWER', 'Parameter']
    else:
        normalisation_factor = 1
    
    # Filter elements that correspond to power output
    mask = df.index.str.contains('p\[')
    df_out = df[mask].copy()

    # Extract DUIDs
    df_out['DUID'] = df_out.apply(lambda x: re.findall(r'p\[(.+)\]', x.name)[0], axis=1)

    # Compute power output (scale by 100 to account for p.u. normalisation)
    df_out['value'] = df_out.apply(lambda x: x['Variable']['Value'] * normalisation_factor, axis=1)

    # Join emissions intensities and marginal costs
    df_out = df_out.merge(model_data.df_g[['SRMC_2016-17', 'EMISSIONS', 'NEM_REGION']], left_on='DUID', right_index=True, how='left')

    # Scenario duration (hours)
    scenario_duration = model_data.df_scenarios.loc[('hours', 'duration'), (week, scenario)]

    # Regional and national scenario revenue
    scenario_revenue = df_out.groupby('NEM_REGION').apply(lambda x: (x['EMISSIONS'] - baseline) * permit_price * x['value']).reset_index().groupby('NEM_REGION')[0].sum().mul(scenario_duration)
    scenario_revenue.loc['NATIONAL'] = scenario_revenue.sum()
    
    # Regional and national scenario emissions
    scenario_emissions = df_out.groupby('NEM_REGION').apply(lambda x: x['EMISSIONS'] * x['value']).reset_index().groupby('NEM_REGION')[0].sum().mul(scenario_duration)
    scenario_emissions.loc['NATIONAL'] = scenario_emissions.sum()
    
    # Regional and national scenario emissions intensity
    scenario_emissions_intensity = scenario_emissions / model_data.df_aggregate_scenario_energy_demand.loc[:, (week, scenario)]
    
    # Regional and national scenario emissions intensity of participating generators
    scenario_participating_generators_emissions_intensity = scenario_emissions / (model_data.df_aggregate_scenario_energy_demand.loc[:, (week, scenario)] - model_data.df_aggregate_scenario_hydro_energy_output.loc[:, (week, scenario)] - model_data.df_aggregate_scenario_intermittent_energy_output.loc[:, (week, scenario)])

    # Place results in dictionary
    scenario_revenue_and_emissions = {'scenario_rep_revenue': scenario_revenue, 'scenario_emissions': scenario_emissions, 'scenario_emissions_intensity': scenario_emissions_intensity, 'scenario_participating_generators_emissions_intensity': scenario_participating_generators_emissions_intensity}
    
    return scenario_revenue_and_emissions


# Function to run policy scenario.

# In[10]:


def run_policy_scenario(model, gain, permit_price, baseline_initial, revenue_initial, revenue_target, policy_scenario_type):
    """Run scenario using specified permit prices, gains, baselines, and revenue targets
    
    Params
    ------
    model : Pyomo object
        Instantiated Pyomo object for which parameters will be updated in each scenario
    
    gain : float
        Gain to be used when updating emissions intensity baseline based on difference
        between rolling and target revenue.
        
    permit_price : float
        Emissions permit price [$/tCO2]
        
    baseline_initial : float
        Emissions intensity baseline to use during first week of model run
        
    revenue_initial : float
        Initial scheme revenue endowment [$]
        
    revenue_target : float
        Target scheme revenue [$]
    
    policy_scenario_type : str
        Indentifier for type of policy analysis being investigated. String will appear
        in file names    
    """
       
    # String summarising parameters used in policy scenario analysis
    test_overview_string = """
    Policy Scenario Overview
    ------------------------
    gain: {0}
    permit price: {1}
    baseline initial: {2}
    revenue_initial: {3}
    revenue_target: {4}
    policy_scenario_type: {5}    
    """.format(gain, permit_price, baseline_initial, revenue_initial, revenue_target, policy_scenario_type)
    print(test_overview_string)
    
    # Weeks to loop through
    weeks = range(1, 53)

    # Scenario indices to loop through
    scenarios = range(1, 11)

    # Container for scenario results for each week
    scenario_results = dict()

    # Container for summarised weekly results
    weekly_summary = dict()
    
    # Rolling revenue, updated each iteration, initialised to 0 for each gain scenario
    revenue_rolling = revenue_initial
    
    # Loop through weeks
    for week in weeks:
        # Start clock for iteration time
        time_start = time.time()

        # Initialise results dictionary
        scenario_results[week] = dict()

        # Set emissions intensity baseline if in first period
        if week == 1:
            baseline = baseline_initial

        # Loop through operating scenarios for each week
        for scenario in scenarios:
            # Solve model
            df = solve_scenario(model=model, use_pu=True, permit_price=permit_price, baseline=baseline, week=week, scenario=scenario)
            
            # Get revenue and emissions results
            scenario_revenue_and_emissions = get_scenario_revenue_and_emissions(df, use_pu=True, baseline=baseline, permit_price=permit_price, week=week, scenario=scenario)

            # Get average prices
            scenario_average_electricity_price = get_average_electricity_price(df, use_pu=True, week=week, scenario=scenario)

            # Save results in dictionary
            scenario_results[week][scenario] = {**scenario_revenue_and_emissions,
                                                **scenario_average_electricity_price, 
                                                'gain': gain,
                                                'permit_price': permit_price,
                                                'baseline_initial': baseline_initial, 
                                                'revenue_initial': revenue_initial, 
                                                'revenue_target': revenue_target, 
                                                'policy_scenario_type': policy_scenario_type}

        # End-of-week net revenue from REP payments
        revenue_net_end_of_week = sum(scenario_results[week][scenario]['scenario_rep_revenue'].loc['NATIONAL'] for scenario in scenarios)

        # Rolling scheme revenue (total)
        revenue_rolling += revenue_net_end_of_week

        # Difference between total revenue and target
        revenue_difference = revenue_target - revenue_rolling

        # End-of-week total energy demand
        total_energy_demand_end_of_week = model_data.df_aggregate_scenario_energy_demand.loc['NATIONAL', (week, slice(None))].sum()

        # Total intermittent generation power output
        total_intermittent_energy_end_of_week = model_data.df_aggregate_scenario_intermittent_energy_output.loc['NATIONAL', (week, slice(None))].sum()
        
        # Total hydro power output
        total_hydro_energy_end_of_week = model_data.df_aggregate_scenario_hydro_energy_output.loc['NATIONAL', (week, slice(None))].sum()
        
        # End-of-week total emissions
        emissions_end_of_week = sum(scenario_results[week][scenario]['scenario_emissions'].loc['NATIONAL'] for scenario in scenarios)

        # End-of-week average emissions intensity
        emissions_intensity_end_of_week = emissions_end_of_week / total_energy_demand_end_of_week

        # End-of-week average emissions intensity of generators participating in REP scheme
        emissions_intensity_participating_generators_end_of_week = emissions_end_of_week / (total_energy_demand_end_of_week - total_intermittent_energy_end_of_week - total_hydro_energy_end_of_week)
        
        # Store scenario results in dictionary
        weekly_summary[week] = {'baseline': baseline,
                                'revenue_net_end_of_week': revenue_net_end_of_week,
                                'revenue_rolling': revenue_rolling,
                                'revenue_difference': revenue_difference,
                                'total_energy_demand_end_of_week': total_energy_demand_end_of_week,
                                'emissions_end_of_week': emissions_end_of_week,
                                'emissions_intensity_end_of_week': emissions_intensity_end_of_week,
                                'emissions_intensity_participating_generators_end_of_week': emissions_intensity_participating_generators_end_of_week,
                                'gain': gain,
                                'permit_price': permit_price,
                                'baseline_initial': baseline_initial,
                                'revenue_initial': revenue_initial,
                                'revenue_target': revenue_target,
                                'policy_scenario_type': policy_scenario_type}

        # Update emissions intensity baseline for following week
        if permit_price == 0:
            # If permit price is 0 (i.e. business-as-usual scenario) to prevent baseline from being undefined
            baseline = 0
        else:
            baseline += - ((gain * revenue_difference) / (permit_price * total_energy_demand_end_of_week))

        # Set lower-bound on emissions intensity baseline
        if baseline < 0:
            baseline = 0

        print('Week {0} completed in {1}s.'.format(week, time.time() - time_start))

    # Construct filename using policy scenario type string as an identifer, as well as a timestamp
    scenario_results_filename = 'scenario_results_{0}_{1}.pickle'.format(policy_scenario_type, int(time.time()))
    with open(os.path.join(paths.output_dir, scenario_results_filename), 'wb') as f:
        pickle.dump(scenario_results, f)

    # Construct filename using policy scenario type string as an identifer, as well as a timestamp
    weekly_summary_filename = 'weekly_summary_{0}_{1}.pickle'.format(policy_scenario_type, int(time.time()))
    with open(os.path.join(paths.output_dir, weekly_summary_filename), 'wb') as g:
        pickle.dump(weekly_summary, g)


# ## Run different policy scenarios

# In[11]:


# Business-as-usual
run_policy_scenario(model=model, gain=0, permit_price=0, baseline_initial=0, revenue_initial=0, revenue_target=0, policy_scenario_type='bau')

# Carbon tax
run_policy_scenario(model=model, gain=0, permit_price=40, baseline_initial=0, revenue_initial=0, revenue_target=0, policy_scenario_type='carbon_tax')

# Gain sensitivity analysis
for gain in np.linspace(0, 1, 9):
    run_policy_scenario(model=model, gain=gain, permit_price=40, baseline_initial=1, revenue_initial=0, revenue_target=0, policy_scenario_type='gain_sensitivity_analysis')


# ## Save data

# In[12]:


with open(os.path.join(paths.output_dir, 'df_aggregate_scenario_energy_demand.pickle'), 'wb') as f:
    pickle.dump(model_data.df_aggregate_scenario_energy_demand, f)

