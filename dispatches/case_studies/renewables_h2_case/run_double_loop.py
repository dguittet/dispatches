#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2022 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#
#################################################################################
from pathlib import Path
import os
import json
import pandas as pd
import pyomo.environ as pyo
from pyomo.common.fileutils import this_file_dir

from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds, log_close_to_bounds

from idaes.apps.grid_integration.model_data import ThermalGeneratorModelData
from idaes.apps.grid_integration import Tracker
from dispatches.case_studies.renewables_h2_case.wind_battery_hydrogen_flowsheet import wind_battery_hydrogen_optimize
from dispatches.case_studies.renewables_h2_case.wind_battery_hydrogen_double_loop import MultiPeriodWindBatteryHydrogen
from dispatches.case_studies.renewables_h2_case.re_h2_parameters import re_h2_parameters, get_gen_outputs_from_rtsgmlc, kg_to_tons, h2_mols_per_kg
from dispatches.case_studies.renewables_h2_case.double_loop_operations import soc_target, heuristic_follow_dispatch, dtree

re_h2_dir = Path(this_file_dir())

##################
#
# Setup Scenario
#
##################

wind_gen = "317_WIND"
wind_gen_pmax = 799.1
gas_gen = "317_CT"
gas_gen_pmax = 110
reserves = 15
shortfall = 500
start_date = '2020-01-01 00:00:00'
df = pd.read_csv(re_h2_dir / "data" / "Wind_Thermal_Gen.csv", index_col="Datetime", parse_dates=True)
wind_cfs, wind_resource, loads_mw = get_gen_outputs_from_rtsgmlc(wind_gen, gas_gen, reserves, shortfall, start_date)

dispatch_strategy = "tank_target"        # "tank_target", "discharge_tank", "discharge_batt", "min_op_cost", "dtree"
design = "batth2"
op_setting = "minsoc"                # "freeop", "modop", "minsoc"

horizon = 24 if dispatch_strategy == "min_op_cost" else 1
    
results_dir = re_h2_dir / f"double_loop_{gas_gen}_{design}"

if design == "batth2":
    with open(results_dir / "input_parameters.json", "r") as f:
        params = json.load(f)
    hybrid_wind_mw = params["wind_mw"]
    hybrid_batt_mw = params['batt_mw']
    hybrid_turb_mw = params['turb_mw']
elif design == "batt":
    hybrid_wind_mw = 932.301
    hybrid_batt_mw = 106.314
    hybrid_batt_mwh = 3694.845
    hybrid_pem_mw = 0
    hybrid_tank_tonH2 = 0
    hybrid_turb_mw = 0
    hybrid_turb_conv = 10
    h2_price_per_kg = 2.0

hybrid_pmax = wind_gen_pmax + hybrid_batt_mw + hybrid_turb_mw
n_time_points = len(wind_cfs)



#########################
#
# Run Conceptual Design
#
#########################

params["tank_size"] = params['tank_tonH2'] / kg_to_tons
params['pem_bar'] = re_h2_parameters['pem_bar']

params["wind_resource"] = wind_resource
params["load"] = loads_mw
params["shortfall_price"] = shortfall
params["design_opt"] = params["build_add_wind"] = True
if op_setting == "minsoc":
    params['min_tank_soc'] = hybrid_turb_mw
    params['min_batt_soc'] = gas_gen_pmax - hybrid_turb_mw
    params["modop"] = True
if op_setting == "modop":
    params["modop"] = True

if not results_dir.exists():
    os.mkdir(results_dir)

if (results_dir / f"design_timeseries_{op_setting}.csv").exists():
    res_df = pd.read_csv(results_dir / f"design_timeseries_{op_setting}.csv")
    with open(results_dir / f"design_sizes_{op_setting}.json", 'r') as f:
        des_res = json.load(f)
else:
    des_res, res_df = wind_battery_hydrogen_optimize(n_time_points=n_time_points, input_params=params, verbose=False, plot=False)
    res_df.to_csv(results_dir / f"design_timeseries_{op_setting}.csv")
    with open(results_dir / f"design_sizes_{op_setting}.json", 'w') as f:
        json.dump(des_res, f)

params.update(des_res)
params["tank_size"] = params['tank_tonH2'] / kg_to_tons

##########################
#
# Setup Operation Model
#
##########################

params['tank_holdup_init_mols'] = res_df["Tank Holdup [kg]"].values[-1] * h2_mols_per_kg
params['batt_soc_init_mwh'] = res_df["State of Charge [MWh]"].values[-1]
wind_gen_pmax = params['wind_mw']
hybrid_pmax = params['wind_mw'] + params['turb_mw'] + params['batt_mw']

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
    tracking_horizon=horizon,
    n_tracking_hour=1,
    solver=solver,
)

blk = tracker_object.model.fs
active_blks = tracker_object.model.fs.windBatteryHydrogen.get_active_process_blocks()
fs = tracker_object.model.fs.windBatteryHydrogen

profiles = {
    "realized_soc": [params['batt_soc_init_mwh'] * 1e3], 
    "realized_energy_throughput": [pyo.value(active_blks[0].fs.battery.initial_energy_throughput)],
    "realized_h2_tank_holdup": [params['tank_holdup_init_mols']]
    }
# profiles = {'realized_soc': ([839887.7]), 'realized_energy_throughput': ([629573.34]), 'realized_h2_tank_holdup': ([128232983.45894864])}
# profiles = {'realized_soc': ([839887.61]), 'realized_energy_throughput': ([1401265.58]), 'realized_h2_tank_holdup': ([462870.2998072651])}


############################
#
# Operation Strategy Setup
#
############################

if dispatch_strategy == "discharge_tank":
    dispatch_strategy_fx = heuristic_follow_dispatch
    params['discharge_first'] = "tank"
elif dispatch_strategy == 'discharge_batt':
    dispatch_strategy_fx = heuristic_follow_dispatch
    params['discharge_first'] = "batt"
elif "target" in dispatch_strategy:
    dispatch_strategy_fx = soc_target
    blk.del_component("tot_cost")
    blk.tot_cost = pyo.Expression(blk.HOUR)
    blk.tank_target = pyo.Param(blk.HOUR, mutable=True)
    blk.tank_target_under = pyo.Var(blk.HOUR, initialize=0, within=pyo.NonNegativeReals)
    blk.tank_target_over = pyo.Var(blk.HOUR, initialize=0, within=pyo.NonNegativeReals)
    blk.tank_target_constraint = pyo.Constraint(blk.HOUR,  rule=lambda b, t: 
                                    active_blks[t].fs.h2_tank.tank_holdup[0] + b.tank_target_under[t] == b.tank_target[t] + b.tank_target_over[t])
    for (t, b) in enumerate(active_blks):
        blk.tot_cost[t] = (blk.tank_target_under[t] + blk.tank_target_over[t]) * 1000

    tracker_object.model.del_component('obj')
    tracker_object._add_tracking_objective()

elif dispatch_strategy == "min_op_cost":
    dispatch_strategy_fx = lambda tracker, params, dispatch, profiles, target_profiles, verbose=None: None
elif dispatch_strategy == 'dtree':
    dispatch_strategy_fx = dtree
    res_df["Load [MW]"] = loads_mw
    res_df["Wind Max [MW]"] = params['wind_mw'] * wind_cfs
    params['year_results'] = res_df


##########################
#
# Operation Optimization
#
##########################

for n, datetime in enumerate(df.index):
    # if n < 4912:
        # profiles = {'realized_soc': ([6530.640934472234]), 'realized_energy_throughput': ([1734298.9265561257]), 'realized_h2_tank_holdup': ([50.0])}
        # tracker_object.update_model(**profiles)
        # continue
    dispatch = mp_model._design_params['load'][n : n + horizon]
    if not len(dispatch):
        break
    date, hour = str(datetime).split(' ')
    print(n, date, hour)

    target_profiles = {
        "batt_kwh": res_df['State of Charge [MWh]'].values[n : n + horizon] * 1e3,
        "tank_holdup_mol": res_df['Tank Holdup [kg]'].values[n : n + horizon] * h2_mols_per_kg,
        "timestep": n
    }

    dispatch_strategy_fx(tracker=tracker_object, params=params, dispatch=dispatch, profiles=profiles, target_profiles=target_profiles)

    try:
        profiles = tracker_object.track_market_dispatch(dispatch, date, hour)
    except Exception as e:
        print(n, profiles)
        dispatch_strategy_fx(tracker=tracker_object, dispatch=dispatch, profiles=profiles, target_profiles=target_profiles, verbose=True)
        profiles = tracker_object.track_market_dispatch(dispatch, date, hour)
    # tracker_object.record_results(date=date, hour=hour)
    try:
        tracker_object.update_model(**profiles)
    except Exception as e:
        print(e)
        print(e.args)
        break


#################
#
# Write Outputs
#
#################

operation_results_dir = results_dir / f"{dispatch_strategy}_{horizon}_{op_setting}"

if not operation_results_dir.exists():
    os.mkdir(operation_results_dir)

tracker_object.write_results(operation_results_dir)

tracker_model_df = pd.read_csv(operation_results_dir/"tracking_model_detail.csv")
tracker_model_df = tracker_model_df[tracker_model_df["Horizon [hr]"] == 0]
tracker_df = pd.read_csv(operation_results_dir/"tracker_detail.csv")
tracker_df = tracker_df[tracker_df["Horizon [hr]"] == 0]


n = len(tracker_model_df)
((wind_cfs * hybrid_wind_mw)[0:n] - res_df['Wind Power Output [MW]'][0:n]).min()
((wind_cfs * hybrid_wind_mw)[0:n] - tracker_model_df['Wind Power Output [MW]']).min()

(wind_cfs * hybrid_wind_mw)[0:n] - res_df['Wind Power Output [MW]'][0:n]
tracker_model_df['Wind Power Output [MW]'] - (wind_cfs * hybrid_wind_mw)[0:n]
tracker_model_df['Wind Power Output [MW]'].values - res_df['Wind Power Output [MW]'].values[0:n]
tracker_model_df['Total Power Output [MW]'].values - res_df['Total Power Output [MW]'].values[0:n]
