def get_formatted_weekly_input(self, weekly_input, forecast_intervals=1):
    """Format weekly targets / indicator dictionaries so they can be used in revenue rebalancing or MPC updates

    Parameters
    ----------
    weekly_input : dict
        Target or indicator given corresponds to a given week

    forecast_intervals : int
        Number of forecast intervals if using MPC. Default = 1.

    Returns
    -------
    intermittent_generators_regulated : dict
        Formatted dictionary denoting whether renewables are eligible in following periods.
    """

    # Length of model horizon
    model_horizon = len(weekly_input)

    # Formatted so can be used in either revenue re-balancing or MPC update
    formatted_weekly_input = {i: {j + 1: weekly_input[j + i] for j in range(0, self.forecast_intervals)} for i in range(1, self.model_horizon + 1 - self.forecast_intervals + 1)}

    return formatted_weekly_input
