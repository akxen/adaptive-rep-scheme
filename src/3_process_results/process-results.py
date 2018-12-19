
# coding: utf-8

# # Process Results
# Process model results and create figures.
# 
# ## Import packages

# In[1]:


import os
import re
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# ## Paths

# In[2]:


# Scenarios directory
scenarios_dir = os.path.join(os.path.curdir, os.path.pardir, '1_create_scenarios', 'output')

# Results directory
results_dir = os.path.join(os.path.curdir, os.path.pardir, '2_updating_strategy', 'output')

# Output directory
output_dir = os.path.join(os.path.curdir, 'output')


# ## Import data

# Model data

# In[3]:


with open(os.path.join(results_dir, 'df_aggregate_scenario_energy_demand.pickle'), 'rb') as f:
    df_aggregate_scenario_energy_demand = pickle.load(f)


# Scenario results

# In[4]:


def get_aggregated_scenario_summary_results():
    "Aggregated scenario summary results"
    
    def get_average_weekly_price(data, week, region):
        "Compute average weekly electricity price from scenario data"

        # Total weekly revenue
        weekly_revenue = sum(data[week][scenario]['average_price'].loc[region] * df_aggregate_scenario_energy_demand.loc[region, (week, scenario)] for scenario in range(1, 11))

        # Total weekly energy demand
        weekly_energy_demand = df_aggregate_scenario_energy_demand.loc[region, (week, slice(None))].sum()

        # Average price
        average_price = weekly_revenue / weekly_energy_demand

        return average_price


    # Files containing scenario data
    filenames = [f for f in os.listdir(results_dir) if 'scenario_results' in f]

    # Container for parsed scenario DataFrames
    scenarios = list()

    # Loop through files containing scenario data
    for f in filenames:
        # Open file
        with open(os.path.join(results_dir, f), 'rb') as g:
            data = pickle.load(g)

        # Container to summarise scenario data
        scenario_summary = dict()

        # Loop through weeks and extract data
        for week in range(1, 53):
            scenario_summary[week] = {'average_national_price': get_average_weekly_price(data, week, 'NATIONAL'),
                                      'policy_scenario_type': data[week][1]['policy_scenario_type'],
                                      'gain' : data[week][1]['gain']}

        # Convert to DataFrame    
        df = pd.DataFrame.from_dict(scenario_summary).T

        # Append to container
        scenarios.append(df)

    # Concatenate scenario data DataFrames
    df_s = pd.concat(scenarios)
    
    # Reset index
    df_s = df_s.reset_index().rename(columns={'index': 'week'})
    
    return df_s

df_s = get_aggregated_scenario_summary_results()


# Weekly summary results

# In[5]:


def get_weekly_summary_results():
    "Extract weekly summary information"
    
    # Filename for weekly summary files
    filenames = [f for f in os.listdir(results_dir) if 'weekly_summary' in f]

    # Container for weekly summary results
    results = list()

    # Open each file, convert to DataFrame, and store in list
    for f in filenames:
        # Open file
        with open(os.path.join(results_dir, f), 'rb') as g:
            wk = pickle.load(g)

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(wk).T

        # Append to list holding result summaries for each week
        results.append(df)    

    # Concatenate all weekly summary data in single DataFrame
    df = pd.concat(results, sort=True)
    df = df.reset_index().rename(columns={'index': 'week'})
    
    return df
df_wks = get_weekly_summary_results()

# Merge price data from scenario results processing
df_wks = pd.merge(df_wks, df_s, left_on=['week', 'gain', 'policy_scenario_type'], right_on=['week', 'gain', 'policy_scenario_type'], how='left')


# Plot emissions intensity baseline and rolling scheme revenue for different gain scenarios.

# In[7]:


mask_gain_scenarios = df_wks['policy_scenario_type'] == 'gain_sensitivity_analysis'
gains = df_wks.loc[mask_gain_scenarios, 'gain'].unique()

for gain in gains:

    plt.clf()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Data to plot
    mask_gain_sensitivity_analysis = (df_wks['policy_scenario_type'] == 'gain_sensitivity_analysis') & (df_wks['gain'] == gain)
    x = df_wks.loc[mask_gain_sensitivity_analysis, 'week'].tolist()
    y1 = df_wks.loc[mask_gain_sensitivity_analysis, 'baseline'].tolist()
    y2 = df_wks.loc[mask_gain_sensitivity_analysis, 'emissions_intensity_participating_generators_end_of_week'].tolist()
    y3 = df_wks.loc[mask_gain_sensitivity_analysis, 'revenue_rolling'].mul(1e-6).tolist()

    # Plotting lines
    ln_1 = ax1.plot(x, y1, color='#db1818', label='Baseline')
    ln_2 = ax1.plot(x, y2, color='#118dd6', label='Emissions intensity')
    ln_3 = ax2.plot(x, y3, color='#d8a517', label='Revenue')

    # Legend
    lns = ln_1 + ln_2 + ln_3
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, ncol=3, loc=0, bbox_to_anchor=(1, 1.13))
    
    # Format axes
    ax1.set_ylabel('Baseline / Emissions intensity \n(tCO$_{2}$/MWh)')
    ax2.set_ylabel('Revenue ($10^{6}$\$)')
    ax1.set_xlabel('Week')
    plt.text(0.45, 0.93, 'Gain: {}'.format(gain), weight='bold', transform=ax1.transAxes)

    ax1.minorticks_on()
    ax2.minorticks_on()
    ax1.grid(linestyle='-.')
    ax1.set_ylim([0.87, 1.12])
    
    if gain != 0:
        ax2.set_ylim([-15, 15])
        
    width = 5.95114
    height = width / 1.6
    fig.set_size_inches(width, height)
    fig.subplots_adjust(left=0.14, bottom=0.12, right=0.9, top=0.9)
    
    fig.savefig(os.path.join(output_dir, 'figures', 'emissions_intensity_revenue_gain_{}.pdf'.format(gain)))

    plt.show()


# In[8]:


gain = 0.125

for gain in gains:
    plt.clf()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Business-as-usual price
    mask_bau = (df_wks['policy_scenario_type'] == 'bau')
    x = df_wks.loc[mask_bau, 'week'].tolist()
    y1 = df_wks.loc[mask_bau, 'average_national_price'].tolist()

    # Price for different gains
    mask_gain_sensitivity_analysis = (df_wks['policy_scenario_type'] == 'gain_sensitivity_analysis') & (df_wks['gain'] == gain)
    x = df_wks.loc[mask_gain_sensitivity_analysis, 'week'].tolist()
    y2 = df_wks.loc[mask_gain_sensitivity_analysis, 'average_national_price'].tolist()

    # Emissions intensity baseline
    y3 = df_wks.loc[mask_gain_sensitivity_analysis, 'baseline'].tolist()

    ln_1 = ax2.plot(x, y1, color='#216f8c', label='BAU')
    ln_2 = ax2.plot(x, y2, color='#c18838', label='REP')
    ln_3 = ax1.plot(x, y3, color='#db1818', label='Baseline')

    # Legend
    lns = ln_1 + ln_2 + ln_3
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, ncol=3, loc=0, bbox_to_anchor=(0.9, 1.13))

    # Format axes
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Baseline (tCO$_{2}$/MWh)')
    ax2.set_ylabel('Average price ($/MWh)')
    plt.text(0.45, 0.93, 'Gain: {}'.format(gain), weight='bold', transform=ax1.transAxes)
    
    
    ax1.grid(linestyle='-.')
    ax1.minorticks_on()
    ax2.minorticks_on()
    ax1.set_ylim([0.87, 1.12])
    
    width = 5.95114
    height = width / 1.6
    fig.set_size_inches(width, height)
    fig.subplots_adjust(left=0.115, bottom=0.14, right=0.89, top=0.9)
    
    fig.savefig(os.path.join(output_dir, 'figures', 'average_price_baseline_gain_{}.pdf'.format(gain)))

    plt.show()

