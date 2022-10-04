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
import copy
from pathlib import Path
import os
import json
from functools import partial
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.common.fileutils import this_file_dir

import idaes.logger as idaeslog
from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds, log_close_to_bounds

from idaes.apps.grid_integration.model_data import ThermalGeneratorModelData
from idaes.apps.grid_integration import Tracker
from dispatches.case_studies.renewables_case.RE_flowsheet import create_model, propagate_state, value
from dispatches.case_studies.renewables_h2_case.wind_battery_hydrogen_flowsheet import wind_battery_hydrogen_optimize
from dispatches.case_studies.renewables_h2_case.wind_battery_hydrogen_double_loop import MultiPeriodWindBatteryHydrogen
from dispatches.case_studies.renewables_h2_case.re_h2_parameters import re_h2_parameters, get_gen_outputs_from_rtsgmlc, kg_to_tons, h2_mols_per_kg

re_h2_dir = Path(this_file_dir())

##################
#
# Setup Scenario
#
##################

wind_gen = "317_WIND"
wind_gen_pmax = 799.1
gas_gen = "317_CT"
reserves = 15
shortfall = 500
start_date = '2020-01-01 00:00:00'
df = pd.read_csv(re_h2_dir / "data" / "Wind_Thermal_Gen.csv", index_col="Datetime", parse_dates=True)
wind_cfs, wind_resource, loads_mw = get_gen_outputs_from_rtsgmlc(wind_gen, gas_gen, reserves, shortfall, start_date)

dispatch_strategy = "discharge_batt"        # "discharge_tank", "tank_target", "discharge_batt", "min_op_cost"
design = "batth2"
horizon = 1
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

if not results_dir.exists():
    os.mkdir(results_dir)

if (results_dir / "design_timeseries.csv").exists():
    res_df = pd.read_csv(results_dir / "design_timeseries.csv")
    with open(results_dir / "design_sizes.json", 'r') as f:
        des_res = json.load(f)
else:
    des_res, res_df = wind_battery_hydrogen_optimize(n_time_points=n_time_points, input_params=params, verbose=False, plot=False)
    res_df.to_csv(results_dir / "design_timeseries.csv")
    with open(results_dir / "design_sizes.json", 'w') as f:
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

########################
#
# Operation Heuristics
#
########################

def soc_target(tracker, dispatch, profiles, target_profiles, verbose=False):
    batt_target = target_profiles['batt_kwh'][-1]
    tank_target = target_profiles['tank_holdup_mol'][-1]

    m = active_blks[-1]

    # blk.batt_target[0].set_value(batt_target)
    blk.tank_target[0].set_value(tank_target)

    global total_missed_target
    total_missed_target += pyo.value(blk.tank_target_under[t] + blk.tank_target_over[t])
    # total_missed_target += pyo.value(blk.batt_target_under[t] + blk.batt_target_over[t])
    print(batt_target, tank_target, total_missed_target)


def heuristic_follow_dispatch(discharge_first, dispatch, profiles, target_profiles, verbose=False):
    wind_gen_max = [pyo.value(b.fs.windpower.system_capacity * b.fs.windpower.capacity_factor[0]) for b in active_blks]
    wind_diff = wind_gen_max - dispatch * 1e3
    excess_wind = np.clip(wind_diff, 0, None)
    storage_gen = np.clip(-wind_diff, 0, None)

    batt_soc_max = params['batt_mwh'] * 1e3
    # (fs.pyomo_model.blocks[0].process.fs.battery.nameplate_energy - fs.pyomo_model.blocks[0].process.fs.battery.degradation_rate*fs.pyomo_model.blocks[0].process.fs.battery.energy_throughput[0.0])
    batt_soc = profiles['realized_soc'][0]
    batt_energy_max = params['batt_mw'] * 1e3

    tank_energy_charge_max = params['tank_size'] * 54.517
    tank_energy_charged = profiles['realized_h2_tank_holdup'][0] / h2_mols_per_kg * 54.517
    tank_energy_discharge_max = profiles['realized_h2_tank_holdup'][0] / h2_mols_per_kg * params['turb_conv']

    turb_energy_max = params['turb_mw'] * 1e3

    for i, process_blk in enumerate(active_blks):
        wind_energy = pyo.value(process_blk.fs.windpower.system_capacity * process_blk.fs.windpower.capacity_factor[0])

        if excess_wind[i] > 0:

            energy_to_battery_ts = min(min(batt_energy_max, batt_soc_max - batt_soc), excess_wind[i])
            if energy_to_battery_ts < 1:
                energy_to_battery_ts = 0
            if energy_to_battery_ts / 0.95 > excess_wind[i]:
                energy_to_battery_ts = excess_wind[i] * 0.95
            excess_wind[i] -= energy_to_battery_ts / 0.95
            batt_soc += energy_to_battery_ts
            
            energy_to_pem_ts = min(min(params['pem_mw'] * 1e3, tank_energy_charge_max - tank_energy_charged), excess_wind[i])
            if energy_to_pem_ts < 1:
                energy_to_pem_ts = 0
            tank_energy_charged += energy_to_battery_ts

            batt_out = 0
            turb_out = 0
            wind_energy -= excess_wind[i]

        elif storage_gen[i] > 0:
            turb_out = min(min(tank_energy_discharge_max, turb_energy_max), storage_gen[i])
            batt_out = min(min(batt_soc, batt_energy_max), storage_gen[i])

            if discharge_first == "tank":
                tank_energy_discharge_max -= turb_out
                if tank_energy_discharge_max < 1:
                    turb_out = max(turb_out - 1, 0)
                storage_gen[i] -= turb_out
                batt_out = min(batt_out, storage_gen[i])
                if batt_out / 0.95 > batt_soc:
                    batt_out = batt_soc * 0.95
                batt_soc -= batt_out / 0.95
                storage_gen[i] -= batt_out
            else:
                if batt_out / 0.95 > batt_soc:
                    batt_out = batt_soc * 0.95
                batt_soc -= batt_out / 0.95
                storage_gen[i] -= batt_out
                turb_out = min(turb_out, storage_gen[i])
                tank_energy_discharge_max -= turb_out
                if tank_energy_discharge_max < 1:
                    turb_out = max(turb_out - 1, 0)
                storage_gen[i] -= turb_out
            energy_to_battery_ts = 0
            energy_to_pem_ts = 0
            if abs(storage_gen[i] > 1e-3):
                print("Missed Dispatched: ", storage_gen[i])
                # raise Exception
        else:
            batt_out = energy_to_battery_ts = energy_to_pem_ts = 0

        outlvl = idaeslog.INFO if verbose else idaeslog.WARNING
        solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO, tag="properties")

        m = process_blk
        m.fs.windpower.electricity[0].fix(wind_energy)
        m.fs.windpower.initialize(outlvl=outlvl)

        propagate_state(m.fs.wind_to_splitter)
        m.fs.splitter.battery_elec[0].fix(energy_to_battery_ts)
        m.fs.splitter.pem_elec[0].fix(energy_to_pem_ts)
        m.fs.splitter.initialize(outlvl=outlvl)

        propagate_state(m.fs.splitter_to_pem)
        m.fs.pem.outlet_state[0].flow_mol.fix()
        m.fs.pem.outlet_state[0].mole_frac_comp.fix()
        m.fs.pem.initialize(outlvl=outlvl)
        m.fs.pem.outlet_state[0].flow_mol.unfix()
        m.fs.pem.outlet_state[0].mole_frac_comp.unfix()

        m.fs.battery.elec_in[0].unfix()
        propagate_state(m.fs.splitter_to_battery)
        m.fs.battery.elec_in[0].fix()
        if abs(batt_soc) > 1e-5:
            m.fs.battery.elec_out[0].fix(batt_out)
        else:
            m.fs.battery.elec_out[0].unfix()
        m.fs.battery.initialize(outlvl=outlvl)

        m.fs.h2_tank.properties_in[0].flow_mol.unfix()
        propagate_state(m.fs.pem_to_tank)
        turb_flow = turb_out / 3600 * h2_mols_per_kg / params['turb_conv']
        m.fs.h2_tank.outlet_to_turbine.flow_mol[0].fix(turb_flow)
        m.fs.h2_tank.outlet_to_pipeline.flow_mol[0].fix(0)
        m.fs.h2_tank.properties_in[0].flow_mol.fix()
        m.fs.h2_tank.properties_in[0].mole_frac_comp.fix()
        m.fs.h2_tank.properties_in[0].component('temperature').fix()
        m.fs.h2_tank.properties_in[0].component('pressure').fix()
        m.fs.h2_tank.initialize(outlvl=outlvl)
        if turb_energy_max < 1:
            m.fs.h2_tank.outlet_to_turbine.flow_mol[0].unfix()

    print(energy_to_battery_ts, batt_out, energy_to_pem_ts)    


##########################
#
# Operation Optimization
#
##########################

if dispatch_strategy == "discharge_tank":
    dispatch_strategy_fx = partial(heuristic_follow_dispatch, discharge_first='tank')
elif dispatch_strategy == 'discharge_batt':
    dispatch_strategy_fx = partial(heuristic_follow_dispatch, discharge_first='batt')
elif "target" in dispatch_strategy:
    dispatch_strategy_fx = soc_target
    blk.del_component("tot_cost")
    blk.tot_cost = pyo.Expression(blk.HOUR)
    # blk.batt_target = pyo.Param(blk.HOUR, mutable=True)
    # blk.batt_target_under = pyo.Var(blk.HOUR, initialize=0, within=pyo.NonNegativeReals)
    # blk.batt_target_over = pyo.Var(blk.HOUR, initialize=0, within=pyo.NonNegativeReals)
    # blk.batt_target_constraint = pyo.Constraint(blk.HOUR, rule=lambda b, t: 
    #                                 active_blks[t].fs.battery.state_of_charge[0] + b.batt_target_under[t] == b.batt_target[t] + b.batt_target_over[t])

    blk.tank_target = pyo.Param(blk.HOUR, mutable=True)
    blk.tank_target_under = pyo.Var(blk.HOUR, initialize=0, within=pyo.NonNegativeReals)
    blk.tank_target_over = pyo.Var(blk.HOUR, initialize=0, within=pyo.NonNegativeReals)
    blk.tank_target_constraint = pyo.Constraint(blk.HOUR,  rule=lambda b, t: 
                                    active_blks[t].fs.h2_tank.tank_holdup[0] + b.tank_target_under[t] == b.tank_target[t] + b.tank_target_over[t])
    for (t, b) in enumerate(active_blks):
        blk.tot_cost[t] = (blk.tank_target_under[t] + blk.tank_target_over[t]) * 1000
        # if "target" == "socs_target":
            # blk.tot_cost[t] += (blk.batt_target_under[t] + blk.batt_target_over[t]) * 1000

    tracker_object.model.del_component('obj')
    tracker_object._add_tracking_objective()

    total_missed_target = 0
elif dispatch_strategy == "min_op_cost":
    dispatch_strategy_fx = lambda dispatch, profiles, target_profiles, verbose=None: None

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
        "tank_holdup_mol": res_df['Tank Holdup [kg]'].values[n : n + horizon] * h2_mols_per_kg
    }

    dispatch_strategy_fx(dispatch=dispatch, profiles=profiles, target_profiles=target_profiles)

    try:
        profiles = tracker_object.track_market_dispatch(dispatch, date, hour)
    except Exception as e:
        print(n, profiles)
        dispatch_strategy_fx(dispatch=dispatch, profiles=profiles, target_profiles=target_profiles, verbose=True)
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

operation_results_dir = results_dir / f"{dispatch_strategy}_{horizon}"

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
