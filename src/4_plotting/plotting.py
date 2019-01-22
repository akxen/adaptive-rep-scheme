
# coding: utf-8

# In[107]:


import os
import pickle

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

mpl.rc('font',family='Times New Roman')


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


# In[192]:


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
    if revenue_target is None:
        revenue_neutral = None
        
    elif revenue_target is not None:
        
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
    if renewables_eligible is None:
        renewables_are_eligible = None
        
    elif renewables_eligible is not None:
        # Check if renewabes are eligible for payments in at least one week
        renewables_are_eligible = any([renewables_eligible[i][j] 
                                       for i in renewables_eligible.keys() 
                                       for j in renewables_eligible[i].keys()])
    
    # Renewables don't receive payments in any week, return False
    else:
        renewables_are_eligible = False
    
    return renewables_are_eligible


# In[165]:


def get_series_data(shock_option, update_mode, revenue_neutral, renewables_eligible, forecast_uncertainty_increment, series_name, description_filter=''):
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
        
    forecast_uncertainty_increment : float
        Scaling factor used to perturb perfect forecast
    
    series_name : str
        Name of series for which data should be extracted
    
    description_filter : str
        Word that must appear in case description. Default = ''. Default will always return
        True.
    
    
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
                               and (x['forecast_uncertainty_increment'] == forecast_uncertainty_increment)
                               and (renewables_become_eligible(x['intermittent_generators_regulated']) == renewables_eligible)
                               and (x['shock_option'] == shock_option) 
                               and (x['update_mode'] == update_mode)
                               and (description_filter in x['description']), axis=1)

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


# In[166]:


# Revenue - neutral
# -----------------
# Baselines
# ---------
# Revenue re-balance update - revenue neutral - no shocks
x1, y1, _ = get_series_data(shock_option='NO_SHOCKS', update_mode='REVENUE_REBALANCE_UPDATE', revenue_neutral=True, renewables_eligible=False, forecast_uncertainty_increment=0.05, series_name='baseline')

# MPC update - revenue neutral - no shocks
x2, y2, _ = get_series_data(shock_option='NO_SHOCKS', update_mode='MPC_UPDATE', revenue_neutral=True, renewables_eligible=False, forecast_uncertainty_increment=0.05, series_name='baseline')


# Scheme revenue
# --------------
# Revenue re-balancing update mechanism - rolling scheme revenue
x3, y3, _ = get_series_data(shock_option='NO_SHOCKS', update_mode='REVENUE_REBALANCE_UPDATE', revenue_neutral=True, renewables_eligible=False, forecast_uncertainty_increment=0.05, series_name='rolling_scheme_revenue_interval_end')

# MPC updating mechanism - rolling scheme revenue
x4, y4, _ = get_series_data(shock_option='NO_SHOCKS', update_mode='MPC_UPDATE', revenue_neutral=True, renewables_eligible=False, forecast_uncertainty_increment=0.05, series_name='rolling_scheme_revenue_interval_end')



# Revenue - target
# ----------------
# Baselines
# ---------
# Revenue re-balance update - revenue target - no shocks
x5, y5, r5 = get_series_data(shock_option='NO_SHOCKS', update_mode='REVENUE_REBALANCE_UPDATE', revenue_neutral=False, renewables_eligible=False, forecast_uncertainty_increment=0.05, series_name='baseline')

# MPC update - revenue neutral - no shocks
x6, y6, _ = get_series_data(shock_option='NO_SHOCKS', update_mode='MPC_UPDATE', revenue_neutral=False, renewables_eligible=False, forecast_uncertainty_increment=0.05, series_name='baseline')


# Scheme revenue
# --------------
# Revenue re-balancing update mechanism - rolling scheme revenue
x7, y7, _ = get_series_data(shock_option='NO_SHOCKS', update_mode='REVENUE_REBALANCE_UPDATE', revenue_neutral=False, renewables_eligible=False, forecast_uncertainty_increment=0.05, series_name='rolling_scheme_revenue_interval_end')

# MPC updating mechanism - rolling scheme revenue
x8, y8, _ = get_series_data(shock_option='NO_SHOCKS', update_mode='MPC_UPDATE', revenue_neutral=False, renewables_eligible=False, forecast_uncertainty_increment=0.05, series_name='rolling_scheme_revenue_interval_end')



# Emissions intensity
# -------------------
# Average emissions intensity of regulated generators (generators eligible for emissions payments)
x9, y9, _ = get_series_data(shock_option='NO_SHOCKS', update_mode='MPC_UPDATE', revenue_neutral=True, renewables_eligible=False, forecast_uncertainty_increment=0.05, series_name='average_emissions_intensity_regulated_generators')


# In[167]:


def create_baseline_revenue_figure(fname, 
                                   revenue_rebalance_baseline_1, 
                                   mpc_update_baseline_1,
                                   revenue_rebalance_revenue_1,
                                   mpc_update_revenue_1,
                                   revenue_rebalance_baseline_2, 
                                   mpc_update_baseline_2,
                                   revenue_rebalance_revenue_2,
                                   mpc_update_revenue_2,
                                   emissions_intensity):
   
    # Revenue rebalancing - first scenario
    x1, y1 = revenue_rebalance_baseline_1
    x2, y2 = mpc_update_baseline_1
    x3, y3 = revenue_rebalance_revenue_1
    x4, y4 = mpc_update_revenue_1
    x5, y5 = revenue_rebalance_baseline_2
    x6, y6 = mpc_update_baseline_2
    x7, y7 = revenue_rebalance_revenue_2
    x8, y8 = mpc_update_revenue_2
    x9, y9 = emissions_intensity

    
    # Colours
    # -------
    mpc_curve = '#e23653'
    reb_curve = '#4f8bea'
    emission_int = '#c6ba71'
    rev_target = '#44433d'

    plt.clf()

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row')

    # Revenue neutral
    axs[0, 0].step(x1, y1, where='post', color=reb_curve, linewidth=1.1) # Revenue re-balance baseline
    axs[0, 0].step(x2, y2, where='post', color=mpc_curve, linewidth=1.1) # MPC update baseline
    axs[0, 0].set_ylim([1, 1.04])

    axs[1, 0].plot(x3, [i/1e6 for i in y3], color=reb_curve, linewidth=1.1) # Revenue rebalance scheme revenue
    axs[1, 0].plot(x4, [i/1e6 for i in y4], color=mpc_curve, linewidth=1.1) # MPC update scheme revenue

    # Revenue target
    axs[0, 1].step(x5, y5, where='post', color=reb_curve, linewidth=1.1) # Revenue rebalance - revenue target - baseline
    axs[0, 1].step(x6, y6, where='post', color=mpc_curve, linewidth=1.1) # MPC update - revenue target - baseline
    l1, = axs[1, 1].plot(x7, [i/1e6 for i in y7], color=reb_curve, label='Revenue rebalancing', linewidth=1.1) # Revenue rebalance - revenue target - scheme revenue
    l2, = axs[1, 1].plot(x8, [i/1e6 for i in y8], color=mpc_curve, label='MPC update', linewidth=1.1) # MPC update - revenue target - scheme revenue
    axs[0, 1].set_ylim([1, 1.04])

    # Emissions intensity
    axs[0, 0].plot(x9, y9, color=emission_int, linewidth=1.1)
    l3, = axs[0, 1].plot(x9, y9, color=emission_int, label='Emissions intensity', linewidth=1.1)

    # Revenue targets
    axs[1, 0].plot([x1[0], x1[-1]], [0, 0], color=rev_target, linestyle='--', linewidth=0.9)
    l4, = axs[1, 1].plot([x9[0], x9[-1]], [10, 10], color=rev_target, linestyle='--', linewidth=0.9, label='Revenue target')


    # Format axes
    # -----------
    # Turn on minor ticks
    axs[0, 0].minorticks_on()
    axs[1, 0].minorticks_on()
    axs[1, 1].minorticks_on()

    # Axes labels
    axs[0, 0].set_ylabel('Baseline (tCO$_\mathdefault{2}$/MWh)', fontsize=8)
    axs[0, 0].set_xlabel('(a)', fontsize=8)
    axs[0, 1].set_xlabel('(b)', fontsize=8)
    axs[1, 0].set_ylabel('Revenue (\$ 10$^\mathdefault{6}$)', fontsize=8)
    axs[1, 0].set_xlabel('(c)\nWeek', fontsize=8)
    axs[1, 0].xaxis.set_tick_params(labelsize=8)
    axs[0, 0].yaxis.set_tick_params(labelsize=8)
    axs[1, 0].yaxis.set_tick_params(labelsize=8)
    axs[1, 1].set_xlabel('(d)\nWeek', fontsize=8)
    axs[1, 1].xaxis.set_tick_params(labelsize=8)

    minorLocator = MultipleLocator(2)
    majorLocator = MultipleLocator(10)

    axs[1, 0].xaxis.set_major_locator(majorLocator)
    axs[1, 0].xaxis.set_minor_locator(minorLocator)

    axs[1, 1].xaxis.set_major_locator(majorLocator)
    axs[1, 1].xaxis.set_minor_locator(minorLocator)

    # Legend
    axs[1, 1].legend([l1, l2, l3, l4], ['Revenue rebalance', 'MPC update', 'Emissions intensity', 'Revenue target'], fontsize=7)

    # Set figure size
    width = 17.8
    height = 8.8
    cm_to_in = 0.393701
    fig.set_size_inches(width*cm_to_in, height*cm_to_in)
    fig.subplots_adjust(left=0.07, bottom=0.135, right=0.99, top=0.98, wspace=0.1)
    plt.show()

    fig.savefig(f'output/{fname}', dpi=400)


# In[168]:


create_baseline_revenue_figure(fname='revenue_neutral_and_revenue_target.png',
                               revenue_rebalance_baseline_1=(x1, y1),
                               mpc_update_baseline_1=(x2, y2),
                               revenue_rebalance_revenue_1=(x3, y3),
                               mpc_update_revenue_1=(x4, y4),
                               revenue_rebalance_baseline_2=(x5, y5),
                               mpc_update_baseline_2=(x6, y6),
                               revenue_rebalance_revenue_2=(x7, y7),
                               mpc_update_revenue_2=(x8, y8),
                               emissions_intensity=(x9, y9))


# In[174]:


# Revenue - neutral
# -----------------
# Baselines
# ---------
# Revenue re-balance update - revenue neutral - no shocks
x1, y1, _ = get_series_data(shock_option='EMISSIONS_INTENSITY_SHOCK', update_mode='REVENUE_REBALANCE_UPDATE', revenue_neutral=True, renewables_eligible=False, forecast_uncertainty_increment=0.05, series_name='baseline', description_filter=' anticipated')

# MPC update - revenue neutral - no shocks
x2, y2, _ = get_series_data(shock_option='EMISSIONS_INTENSITY_SHOCK', update_mode='MPC_UPDATE', revenue_neutral=True, renewables_eligible=False, forecast_uncertainty_increment=0.05, series_name='baseline', description_filter=' anticipated')


# Scheme revenue
# --------------
# Revenue re-balancing update mechanism - rolling scheme revenue
x3, y3, _ = get_series_data(shock_option='EMISSIONS_INTENSITY_SHOCK', update_mode='REVENUE_REBALANCE_UPDATE', revenue_neutral=True, renewables_eligible=False, forecast_uncertainty_increment=0.05, series_name='rolling_scheme_revenue_interval_end', description_filter=' anticipated')

# MPC updating mechanism - rolling scheme revenue
x4, y4, _ = get_series_data(shock_option='EMISSIONS_INTENSITY_SHOCK', update_mode='MPC_UPDATE', revenue_neutral=True, renewables_eligible=False, forecast_uncertainty_increment=0.05, series_name='rolling_scheme_revenue_interval_end', description_filter=' anticipated')



# Revenue - target
# ----------------
# Baselines
# ---------
# Revenue re-balance update - revenue target - no shocks
x5, y5, r5 = get_series_data(shock_option='EMISSIONS_INTENSITY_SHOCK', update_mode='REVENUE_REBALANCE_UPDATE', revenue_neutral=True, renewables_eligible=False, forecast_uncertainty_increment=0.05, series_name='baseline', description_filter='unanticipated')

# MPC update - revenue neutral - no shocks
x6, y6, _ = get_series_data(shock_option='EMISSIONS_INTENSITY_SHOCK', update_mode='MPC_UPDATE', revenue_neutral=True, renewables_eligible=False, forecast_uncertainty_increment=0.05, series_name='baseline', description_filter='unanticipated')


# Scheme revenue
# --------------
# Revenue re-balancing update mechanism - rolling scheme revenue
x7, y7, _ = get_series_data(shock_option='EMISSIONS_INTENSITY_SHOCK', update_mode='REVENUE_REBALANCE_UPDATE', revenue_neutral=True, renewables_eligible=False, forecast_uncertainty_increment=0.05, series_name='rolling_scheme_revenue_interval_end', description_filter='unanticipated')

# MPC updating mechanism - rolling scheme revenue
x8, y8, _ = get_series_data(shock_option='EMISSIONS_INTENSITY_SHOCK', update_mode='MPC_UPDATE', revenue_neutral=True, renewables_eligible=False, forecast_uncertainty_increment=0.05, series_name='rolling_scheme_revenue_interval_end', description_filter='unanticipated')



# Emissions intensity
# -------------------
# Average emissions intensity of regulated generators (generators eligible for emissions payments)
x9, y9, _ = get_series_data(shock_option='EMISSIONS_INTENSITY_SHOCK', update_mode='MPC_UPDATE', revenue_neutral=True, renewables_eligible=False, forecast_uncertainty_increment=0.05, series_name='average_emissions_intensity_regulated_generators', description_filter='unanticipated')


# In[175]:


for index, row in run_summaries.iterrows():
    print(index, row['description'])


# In[181]:


run_id = '1496BFFB'
series_name = 'baseline'

def get_case_data(run_id, series_name):
    "Extract data for given run_id and series_name"
    
    with open(os.path.join(results_dir, f'{run_id}_week_metrics.pickle'), 'rb') as f:
        week_metrics = pickle.load(f)

    return list(week_metrics[series_name].keys()), list(week_metrics[series_name].values())


# In[183]:


x, y = get_case_data(run_id, series_name)


# update_mode - REVENUE_REBALANCE_UPDATE, MPC_UPDATE
# shock_type - NO_SHOCKS, EMISSIONS_INTENSITY_SHOCK
# 
# revenue neutral - True / False
# renewables_become_eligible - True / False
# 
# forecast_uncertainty increment - float
# anticipated_shock - True / False / None

# In[212]:


def get_case_run_id(update_mode, shock_option, anticipated_shock, forecast_uncertainty_increment, revenue_neutral, renewables_eligible):
    
    "Given case parameters, find run_id corresponding to case"

    if anticipated_shock is None:
        description_filter_1 = ''
    elif anticipated_shock:
        description_filter_1 = ' anticipated'
    else:
        description_filter_1 = ' unanticipated'


    # run_summaries.loc[(run_summaries['update_mode'] == update_mode) and ]
    mask = run_summaries.apply(lambda x: (x['shock_option'] == shock_option)
                               and (x['update_mode'] == update_mode)
                               and (x['forecast_uncertainty_increment'] == forecast_uncertainty_increment)
                               and (is_revenue_neutral(x['target_scheme_revenue']) == revenue_neutral)
                               and (renewables_become_eligible(x['intermittent_generators_regulated']) == renewables_eligible)
                               and (description_filter_1 in x['description']), axis=1)

    if len(run_summaries.loc[mask].index) != 1:
        raise(Exception(f'Should only return 1 run_id, returned : {run_summaries.loc[mask].index}'))

    return run_summaries.loc[mask].index[0]


update_mode = 'MPC_UPDATE'
shock_option = 'NO_SHOCKS'
forecast_uncertainty_increment = 0.05
revenue_neutral = True
renewables_eligible = True
anticipated_shock = None

get_case_run_id(update_mode='MPC_UPDATE',
                shock_option='NO_SHOCKS',
                anticipated_shock=None,
                forecast_uncertainty_increment=0.05,
                revenue_neutral=True,
                renewables_eligible=True)

