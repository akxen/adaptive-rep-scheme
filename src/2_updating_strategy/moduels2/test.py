import os

from AdaptiveREP import Forecast, Cases

output_dir = os.path.join(os.path.curdir, 'output')

fc = Forecast.ConstructForecast(output_dir)

run_id = '67FE65FC'

a = fc._load_week_metrics(run_id)

b = fc._get_model_horizon(run_id)

c = fc._get_run_summaries()

shock_option = 'NO_SHOCKS'
initial_permit_price = 0
model_horizon = 5
d = fc._get_benchmark_run_id(shock_option, initial_permit_price, model_horizon)

forecast_intervals = 1

e = {1: fc.get_forecast(benchmark_run_id=d, forecast_type='intermittent_energy', forecast_intervals=2, forecast_uncertainty_increment=0.05)}
