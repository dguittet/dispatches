import copy
from pathlib import Path
import os
import pandas as pd
import pyomo.environ as pyo
from pyomo.common.fileutils import this_file_dir

from idaes.apps.grid_integration.model_data import ThermalGeneratorModelData
from idaes.apps.grid_integration import Tracker
from dispatches.case_studies.renewables_h2_case.wind_battery_hydrogen_flowsheet import wind_battery_hydrogen_optimize
from dispatches.case_studies.renewables_h2_case.wind_battery_hydrogen_double_loop import MultiPeriodWindBatteryHydrogen
from dispatches.case_studies.renewables_h2_case.re_h2_parameters import re_h2_parameters, get_gen_outputs_from_rtsgmlc

re_h2_dir = Path(this_file_dir())

params = copy.copy(re_h2_parameters)
wind_gen = "317_WIND"
wind_gen_pmax = 799.1
gas_gen = "317_CT"
reserves = 15
shortfall = 500
start_date = '2020-01-01 00:00:00'
df = pd.read_csv(re_h2_dir / "data" / "Wind_Thermal_Gen.csv", index_col="Datetime", parse_dates=True)
wind_cfs, wind_resource, loads_mw = get_gen_outputs_from_rtsgmlc(wind_gen, gas_gen, reserves, shortfall, start_date)

hybrid_wind_mw = 953.842
hybrid_batt_mw = 94.33
hybrid_batt_mwh = 839.888
hybrid_pem_mw = 3.773
hybrid_tank_tonH2 = 282.705
hybrid_turb_mw = 45.985
hybrid_turb_conv = 10
h2_price_per_kg = 2.0

params["wind_mw"] = wind_gen_pmax
params["batt_mw"] = hybrid_batt_mw
params["batt_mwh"] = hybrid_batt_mwh
params["pem_mw"] = hybrid_pem_mw
params["tank_tonH2"] = hybrid_tank_tonH2
params["turb_mw"] = hybrid_turb_mw
params["turb_conv"] = hybrid_turb_conv

params["wind_resource"] = wind_resource
params["load"] = loads_mw
params["shortfall_price"] = shortfall
params["h2_price_per_kg"] = h2_price_per_kg * 0
params["design_opt"] = False

hybrid_pmax = wind_gen_pmax + params['batt_mw'] + params['turb_mw']
n_time_points = len(wind_cfs)

results_dir = re_h2_dir / f"double_loop_{gas_gen}_{round(hybrid_pmax)}"
if not results_dir.exists():
    os.mkdir(results_dir)

if (results_dir / "design_results.csv").exists():
    res_df = pd.read_csv(results_dir / "design_results.csv")
else:
    des_res, res_df = wind_battery_hydrogen_optimize(n_time_points=n_time_points, input_params=params, verbose=False, plot=False)
    res_df.to_csv(results_dir / "design_results.csv")

generator_params = {
    "gen_name": wind_gen,
    "bus": "Chulsi",
    "p_min": 0,
    "p_max": wind_gen_pmax,
    "min_down_time": 0,
    "min_up_time": 0,
    "ramp_up_60min": hybrid_pmax,
    "ramp_down_60min": hybrid_pmax,
    "shutdown_capacity": hybrid_pmax,
    "startup_capacity": 0,
    "initial_status": 1,
    "initial_p_output": 0,
    "production_cost_bid_pairs": [(0, 0), (wind_gen_pmax, 0)],
    "startup_cost_pairs": [(0, 0)],
    "fixed_commitment": None,
}
model_data = ThermalGeneratorModelData(**generator_params)
mp_model = MultiPeriodWindBatteryHydrogen(
    model_data, wind_cfs, params
)

solver = pyo.SolverFactory("xpress_direct")
tracker_object = Tracker(
    tracking_model_object=mp_model,
    tracking_horizon=24,
    n_tracking_hour=1,
    solver=solver,
)

for n, datetime in enumerate(df.index):
    dispatch = mp_model._design_params['load'][n : n + 24]
    if not len(dispatch):
        break
    date, hour = str(datetime).split(' ')
    profiles = tracker_object.track_market_dispatch(dispatch, date, hour)
    tracker_object.update_model(**profiles)

tracker_object.write_results(results_dir)

tracker_model_df = pd.read_csv(results_dir/"tracking_model_detail.csv")
tracker_df = pd.read_csv(results_dir/"tracker_detail.csv")

n = len(res_df)
((wind_cfs * wind_gen_pmax)[0:n] - res_df['Wind Power Output [MW]']).min()
((wind_cfs * wind_gen_pmax)[0:n] - tracker_model_df['Wind Power Output [MW]']).min()

(wind_cfs * wind_gen_pmax)[0:n] - res_df['Wind Power Output [MW]']
tracker_model_df['Wind Power Output [MW]'] - (wind_cfs * wind_gen_pmax)[0:n]
tracker_model_df['Wind Power Output [MW]'].values - res_df['Wind Power Output [MW]'].values
tracker_model_df['Total Power Output [MW]'].values - res_df['Total Power Output [MW]'].values
