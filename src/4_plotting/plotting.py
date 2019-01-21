
# coding: utf-8

# In[1]:


import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt


# Paths to directories

# In[2]:


# Directory containing results
results_dir = os.path.join(os.path.curdir, os.path.pardir, '2_updating_strategy', 'modules', 'output')


# Get summary of all scenarios for which model has been run.

# In[3]:


def get_run_summaries(results_dir):
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

# Summary of parameters used for each run
run_summaries = pd.DataFrame.from_dict(get_run_summaries(results_dir), orient='index')


# In[4]:


run_summaries


# In[5]:


def is_revenue_neutral(revenue_target):
    """Check if case is revenue neutral
    
    Parameters
    ----------
    revenue_target : dict or None
        Defines revenue target for each period.
        Note: Will be None when running benchmark cases
    
    Returns
    -------
    revenue_neutral : bool
        True = target scheme revenue is 0 for all periods
        False = Non-zero scheme revenue target for at least one period    
    """
    
    # Check if revenue target is None
    if revenue_target is not None:
        
        # Check if revenue target for all periods = 0
        revenue_neutral =  all([True if revenue_target[i][j] == 0 else False 
                                for i in revenue_target.keys() 
                                for j in revenue_target[i].keys()])
    
    # If revenue target is None, return False
    else:
        revenue_neutral = False
    
    return revenue_neutral


def renewables_become_eligible(renewables_eligible):
    """Check if renewables become eligible to receive payments
    
    Parameters
    ----------
    renewables_eligibile : dict or None
        Defines if renewables are eligible for payments in each period
        Note: Will be None when running benchmark cases
    
    Returns
    -------
    renewables_are_eligible : bool
        True = renewables are eligible for payments in at least one week
        False = renewables are ineligible for payments for all weeks    
    """
    
    # Check if input is type None
    if renewables_eligible is not None:
        # Check if renewabes are eligible for payments in at least one week
        renewables_are_eligible = any([renewables_eligible[i][j] 
                                       for i in renewables_eligible.keys() 
                                       for j in renewables_eligible[i].keys()])
    
    # Renewables don't receive payments in any week, return False
    else:
        renewables_are_eligible = False
    
    return renewables_are_eligible


# In[6]:


def get_series_data(shock_option, update_mode, revenue_neutral, renewables_eligible, series_name):
    """Extract data for given case
    
    Parameters
    ----------
    shock_option : str
        Type of shock applied to the scheme
    
    update_mode : str
        Updating mechanism used
    
    revenue_neutral : bool
        Defines if revenue neutral schemes are to be investigated
    
    renewables_eligible : bool
        Defines if renewables are eligible for emissions payments under scheme
    
    series_name : str
        Name of series for which data should be extracted
    
    
    Returns
    -------
    x : list
        list of independent variables
    
    y : list
        list of dependent variables
    
    run_id : str
        ID of case for which data has been extracted
    """

    # Filter case run IDs by following criteria
    mask = run_summaries.apply(lambda x: (is_revenue_neutral(x['target_scheme_revenue']) == revenue_neutral)
                               and (renewables_become_eligible(x['intermittent_generators_regulated']) == renewables_eligible)
                               and (x['shock_option'] == shock_option) 
                               and (x['update_mode'] == update_mode), axis=1)

    # Check number of cases returned (should only be one case matching all criteria)
    if len(run_summaries.loc[mask].index) != 1:
        raise(Exception(f'Returned more than one case matching criteria: {run_summaries.loc[mask].index}'))
    else:
        run_id = run_summaries.loc[mask].index[0]

    # Load week metrics data for given run_id
    with open(os.path.join(results_dir, f'{run_id}_week_metrics.pickle'), 'rb') as f:
        week_metrics = pickle.load(f)

    # Independent variables
    x = pd.DataFrame(week_metrics).index.tolist()
    
    # Dependent variables 
    y = pd.DataFrame(week_metrics)[series_name].tolist()

    return x, y, run_id


# In[25]:


# Baselines
# ---------
# Revenue re-balance update - revenue neutral - no shocks
x1, y1, r_1 = get_series_data(shock_option='NO_SHOCKS', update_mode='REVENUE_REBALANCE_UPDATE', revenue_neutral=True, renewables_eligible=False, series_name='baseline')

# MPC update - revenue neutral - no shocks
x2, y2, _ = get_series_data(shock_option='NO_SHOCKS', update_mode='MPC_UPDATE', revenue_neutral=True, renewables_eligible=False, series_name='baseline')


# Revenue - neutral
# -----------------
# Revenue re-balancing update mechanism - rolling scheme revenue
x3, y3, _ = get_series_data(shock_option='NO_SHOCKS', update_mode='REVENUE_REBALANCE_UPDATE', revenue_neutral=True, renewables_eligible=False, series_name='rolling_scheme_revenue_interval_end')

# MPC updating mechanism - rolling scheme revenue
x4, y4, _ = get_series_data(shock_option='NO_SHOCKS', update_mode='MPC_UPDATE', revenue_neutral=True, renewables_eligible=False, series_name='rolling_scheme_revenue_interval_end')


# 



# Emissions intensity
# -------------------
# Average emissions intensity of regulated generators (generators eligible for emissions payments)
x5, y5, _ = get_series_data(shock_option='NO_SHOCKS', update_mode='MPC_UPDATE', revenue_neutral=True, renewables_eligible=False, series_name='average_emissions_intensity_regulated_generators')


# In[44]:


plt.clf()

fig, axs = plt.subplots(nrows=2, ncols=2)

axs[0, 0].step(x1, y1, where='post')
axs[0, 0].step(x2, y2, where='post')
axs[0, 0].plot(x5, y5, 'r')
axs[0, 0].set_ylim([1, 1.04])

axs[1, 0].plot(x3, y3, 'r')
axs[1, 0].plot(x4, y4, 'g')

plt.show()


# In[29]:


plt.clf()

fig, ax = plt.subplots()


plt.show()


# In[21]:


run_id = r_1
with open(os.path.join(results_dir, f'{run_id}_week_metrics.pickle'), 'rb') as f:
    week_metrics = pickle.load(f)

pd.DataFrame(week_metrics)

