
# coding: utf-8

# Code snippets that may be usefule when processing model results

# In[ ]:


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
    
    
# # Class object used to process scenario results and produce weekly statistics
# WeeklyResults = ProcessScenarioResults(data_dir, scenarios_dir)

# # Average energy price each week - regional and national statistics
# weekly_average_energy_price = WeeklyResults.get_weekly_average_energy_price(scenario_energy_revenue)

# # Average system emissions intensity each week - regional and national statistics
# weekly_system_emissions_intensity = WeeklyResults.get_weekly_system_emissions_intensity(scenario_total_emissions)

# # Average emissions intensity of generators subject to emissions policy - regional and national statistics
# weekly_regulated_generators_emissions_intensity = WeeklyResults.get_weekly_regulated_generators_emissions_intensity(scenario_total_emissions, scenario_generation_by_fuel_type)

# # Generation by fuel type
# weekly_generation_by_fuel_type = WeeklyResults.get_weekly_generation_by_fuel_type(scenario_generation_by_fuel_type)

# plt.clf()
# pd.DataFrame.from_dict({week: {'baseline': baseline} for week, baseline in weekly_baseline.items()}, orient='index').plot()
# plt.show()

# plt.clf()
# pd.DataFrame.from_dict(scenario_scheme_revenue, orient='index').reset_index().sort_index(axis=1).drop('level_1', axis=1).groupby('level_0').sum().cumsum()['NATIONAL'].plot()
# plt.show()


# Used to compute statistics for each scenario

# In[ ]:


#     def get_scenario_total_emissions(self):
#         """Total emissions [tCO2] for each scenario

#         Returns
#         -------
#         scenario_emissions : dict
#             Total emissions for each NEM region and national level for given scenario    
#         """

#         # Power output and emissions
#         df = pd.DataFrame({'p': self.model.p.get_values()}).mul(self.model.BASE_POWER.value).join(self.df_g[['EMISSIONS', 'NEM_REGION', 'FUEL_TYPE']], how='left')

#         # Dictionary in which to store aggregate results
#         emissions_scenario_results = dict()

#         # Compute total emissions in each NEM region per hour
#         df_emissions = df.groupby('NEM_REGION').apply(lambda x: x.prod(axis=1).sum())

#         # Sum to find national total per hour
#         df_emissions.loc['NATIONAL'] = df_emissions.sum()

#         # Multiply by scenario duration to find total emissions for scenario
#         df_emissions = df_emissions.mul(self.df_scenarios.loc[('hours', 'duration'), (self.week_index, self.scenario_index)])

#         # Dictionary containing emissions data
#         scenario_emissions = df_emissions.to_dict()

#         return scenario_emissions
    
    
#     def get_scenario_scheme_revenue(self):
#         """Scheme revenue for given scenario
        
#         Returns
#         -------
#         scenario_scheme_revenue : dict
#             Scheme revenue for each NEM region as well as national total for given scenario        
#         """

#         # Power output and emissions
#         df = pd.DataFrame({'p': self.model.p.get_values()}).mul(self.model.BASE_POWER.value).join(self.df_g[['EMISSIONS', 'NEM_REGION', 'FUEL_TYPE']], how='left')

#         # Scheme revenue for each NEM region per hour
#         df_scheme_revenue = df.groupby('NEM_REGION').apply(lambda x: x['EMISSIONS'].subtract(self.model.PHI.value).mul(x['p']).sum())

#         # Scheme revenue for nation per hour
#         df_scheme_revenue.loc['NATIONAL'] = df_scheme_revenue.sum()

#         # Multiply by scenario duration to get total scheme revenue
#         df_scheme_revenue = df_scheme_revenue.mul(self.df_scenarios.loc[('hours', 'duration'), (self.week_index, self.scenario_index)])

#         # Convert to dictionary
#         scenario_scheme_revenue = df_scheme_revenue.to_dict()
        
#         return scenario_scheme_revenue
    
    
#     def get_scenario_energy_revenue(self):
#         """Revenue obtained from energy sales for given scenario [$]
        
#         Returns
#         -------
#         scenario_energy_revenue : dict
#             Revenue from energy sales for each NEM region as well as national total for given scenario        
#         """
        
#         # Revenue from energy sales
#         # -------------------------
#         df = pd.DataFrame.from_dict({n: [self.model.dual[self.model.POWER_BALANCE[n]] * self.model.BASE_POWER.value] for n in self.model.OMEGA_N}, columns=['price'], orient='index')

#         # Demand for given scenario
#         df_demand = self.df_scenarios.loc[('demand'), (self.week_index, self.scenario_index)]
#         df_demand.name = 'demand'

#         # Total revenue from electricity sales per hour
#         df_energy_revenue = df.join(df_demand).join(self.df_n['NEM_REGION'], how='left').groupby('NEM_REGION').apply(lambda x: x['price'].mul(x['demand']).sum())
#         df_energy_revenue.loc['NATIONAL'] = df_energy_revenue.sum()

#         # Total revenue from energy sales
#         df_energy_revenue = df_energy_revenue.mul(self.df_scenarios.loc[('hours', 'duration'), (self.week_index, self.scenario_index)])

#         # Add to energy revenue results dictionary
#         scenario_energy_revenue = df_energy_revenue.to_dict()
        
#         return scenario_energy_revenue


#     def get_scenario_generation_by_fuel_type(self):
#         """Generation by fuel type for each NEM region

#         Returns
#         -------
#         scenario_generation_by_fuel_type : dict
#             Total energy output [MWh] for each type of generating unit for each NEM region as well as national total
#         """

#         # Power output
#         df = pd.DataFrame({'p': self.model.p.get_values()}).mul(self.model.BASE_POWER.value).join(self.df_g[['EMISSIONS', 'NEM_REGION', 'FUEL_TYPE']], how='left')

#         # Energy output by fuel type for each NEM region [MWh]
#         df_fuel_type_generation = df.groupby(['NEM_REGION', 'FUEL_TYPE'])['p'].sum().mul(self.df_scenarios.loc[('hours', 'duration'), (self.week_index, self.scenario_index)])

#         # National total
#         df_nation = df_fuel_type_generation.reset_index().groupby('FUEL_TYPE')['p'].sum().reset_index()
#         df_nation['NEM_REGION'] = 'NATIONAL'

#         # Combine regional and national values and convert to dictionary
#         scenario_generation_by_fuel_type = pd.concat([df_fuel_type_generation.reset_index(), df_nation], sort=False).set_index(['NEM_REGION', 'FUEL_TYPE'])['p'].to_dict()

#         return scenario_generation_by_fuel_type
    
    
#     def get_scenario_energy_output(self):
#         """Energy output from dispatchable generators [MWh] for each NEM region and national total
        
#         Returns
#         -------
#         total_energy_output : pandas DataFrame
#             Total energy output from all dispatchable generators under the emissions policy       
#         """
        
#         # Power output
#         df = pd.DataFrame({'p': self.model.p.get_values()}).mul(self.model.BASE_POWER.value).join(self.df_g[['NEM_REGION']], how='left')

#         # Energy output by fuel type for each NEM region [MWh]
#         df_energy_output = df.groupby(['NEM_REGION'])['p'].sum().mul(self.df_scenarios.loc[('hours', 'duration'), (self.week_index, self.scenario_index)])

#         # Compute national total
#         df_energy_output.loc['NATIONAL'] = df_energy_output.sum()

#         return df_energy_output


# Find ID of price setting generator

# In[ ]:


new_srmc = (DCOPF.df_g['SRMC_2016-17'] + ((DCOPF.df_g['EMISSIONS'] * permit_price))) / 100
pd.DataFrame.from_dict(scenario_nodal_prices).applymap(lambda x: new_srmc.subtract(x).abs().min()).max().max()


# Compute weekly statistics given at regional and national levels from scenario results

# In[ ]:


# Generator output results
# ------------------------
# Power output from each generator for each time period
df = pd.DataFrame(scenario_power_output)

# Re-organise DataFrame so multi-index is removed, with individual columns containing week and scenario indices
df_o = df.reset_index().melt(id_vars=['index']).rename(columns={'index': 'DUID', 'variable_0': 'week_index', 'variable_1': 'scenario_index', 'value': 'power_pu'})

# Get duration of each scenario
df_o['duration_hrs']= df_o.apply(lambda x: DCOPF.df_scenarios.loc[('hours', 'duration'), (x['week_index'], x['scenario_index'])], axis=1)

# Get emissions, NEM region, and fuel type for each DUID
df_o = pd.merge(df_o, DCOPF.df_g[['EMISSIONS', 'NEM_REGION', 'FUEL_TYPE']], how='left', left_on='DUID', right_index=True)

# Total energy output for each DUID and each scenario
df_o['energy_MWh'] = df_o['power_pu'].mul(100).mul(df_o['duration_hrs'])

# Total emissions for each DUID and each scenario
df_o['emissions_tCO2'] = df_o['energy_MWh'].mul(df_o['EMISSIONS'])


# Scenario information
# --------------------
# Copy original DataFrame
df_s = DCOPF.df_scenarios.copy()

# Duration for each scenario
duration = df_s.loc[('hours', 'duration')]
duration = duration.reset_index()
duration.columns = duration.columns.droplevel(level=1)

# Fixed nodal injections and withdrawals (demand, hydro, and intermittent sources)
df_fixed = df_s.drop(('hours', 'duration')).reset_index().melt(id_vars=['level', 'NODE_ID']).rename(columns={'value': 'power_MW'}).astype({'NODE_ID': int, 'week': int, 'scenario': int})

# Merge duration for each operating scenario
df_fixed = pd.merge(df_fixed, duration, how='left', left_on=['week', 'scenario'], right_on=['week', 'scenario'])

# Compute total energy injection / withdrawal
df_fixed['energy_MWh'] = df_fixed['power_MW'].mul(df_fixed['hours'])

# Merge NEM region IDs
df_fixed = pd.merge(df_fixed, DCOPF.df_n[['NEM_REGION']], how='left', left_on='NODE_ID', right_index=True)

# Aggregate fixed injections for each NEM region
df_reg_fixed = df_fixed.groupby(['NEM_REGION', 'week', 'level'])[['energy_MWh']].sum().reset_index()

# Aggregate fixed injections for whole nation
df_nat_fixed = df_reg_fixed.groupby(['week', 'level'])[['energy_MWh']].sum().reset_index()
df_nat_fixed['NEM_REGION'] = 'NATIONAL'

# Combine regional and national level statistics
df_com_fixed = pd.concat([df_reg_fixed, df_nat_fixed], sort=False).set_index(['NEM_REGION', 'level', 'week']).sort_index()


# Weekly regional and national statistics
# ---------------------------------------
# Total energy output and emissions from regulated generators for each NEM region
df_reg = df_o.groupby(['week_index', 'NEM_REGION'])[['energy_MWh', 'emissions_tCO2']].sum().reset_index()

# Total energy output and emissions from regulated generators for nation
df_nat = df_reg.groupby('week_index')[['energy_MWh', 'emissions_tCO2']].sum().reset_index()
df_nat['NEM_REGION'] = 'NATIONAL'

# Combined regional and national level statistics
df_com = pd.concat([df_reg, df_nat], sort=False).set_index(['NEM_REGION', 'week_index']).sort_index()

# Average emissions intensity of regulated generators
df_com['average_regulated_emissions_intensity'] = df_com['emissions_tCO2'].div(df_com['energy_MWh'])

# Energy demand each period
df_com['demand_MWh'] = df_com.apply(lambda x: df_com_fixed.loc[(x.name[0], 'demand', x.name[1]), 'energy_MWh'], axis=1)

# Fixed energy injections from hydro sources
df_com['hydro_MWh'] = df_com.apply(lambda x: df_com_fixed.loc[(x.name[0], 'hydro', x.name[1]), 'energy_MWh'], axis=1)

# Fixed energy injections from intermittent generators
df_com['intermittent_MWh'] = df_com.apply(lambda x: df_com_fixed.loc[(x.name[0], 'intermittent', x.name[1]), 'energy_MWh'], axis=1)

# Check national energy demand and energy injections balance
df_com['energy_MWh'].add(df_com['hydro_MWh']).add(df_com['intermittent_MWh']).subtract(df_com['demand_MWh']).loc['NATIONAL']

# Average system emissions intensity (total emissions / total demand)
df_com['average_system_emissions_intensity'] = df_com['emissions_tCO2'].div(df_com['demand_MWh'])


# In[1]:


* Total emissions for each NEM region and nation
* Total scheme revenue for each NEM region and nation
* Generation by fuel type and NEM region and nation

Then aggregate scenarios into weekly values
* Average price for each NEM region
* Emissions intensity for system and regulated generators, for nation and each NEM region
* Generation by fuel type and NEM region and nation

