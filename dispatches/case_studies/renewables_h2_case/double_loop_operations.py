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
from sklearn.tree import DecisionTreeRegressor

from idaes.apps.grid_integration.model_data import ThermalGeneratorModelData
from idaes.apps.grid_integration import Tracker
from dispatches.case_studies.renewables_case.RE_flowsheet import create_model, propagate_state, value
from dispatches.case_studies.renewables_h2_case.wind_battery_hydrogen_flowsheet import wind_battery_hydrogen_optimize
from dispatches.case_studies.renewables_h2_case.wind_battery_hydrogen_double_loop import MultiPeriodWindBatteryHydrogen
from dispatches.case_studies.renewables_h2_case.re_h2_parameters import re_h2_parameters, get_gen_outputs_from_rtsgmlc, kg_to_tons, h2_mols_per_kg

########################
#
# Operation Heuristics
#
########################

def soc_target(tracker, params, dispatch, profiles, target_profiles, verbose=False):
    blk = tracker.model.fs
    active_blks = tracker.model.fs.windBatteryHydrogen.get_active_process_blocks()

    batt_target = target_profiles['batt_kwh'][-1]
    tank_target = target_profiles['tank_holdup_mol'][-1]

    m = active_blks[-1]

    # blk.batt_target[0].set_value(batt_target)
    blk.tank_target[0].set_value(tank_target)

    total_missed_target = pyo.value(blk.tank_target_under[0] + blk.tank_target_over[0])
    # total_missed_target += pyo.value(blk.batt_target_under[t] + blk.batt_target_over[t])
    print(batt_target, tank_target, total_missed_target)


def init_with_fixed_controls(process_blk, wind_energy, energy_to_battery_ts, energy_to_pem_ts, batt_soc, batt_out, turb_out, turb_energy_max, turb_conv, verbose):
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
    turb_flow = turb_out / 3600 * h2_mols_per_kg / turb_conv
    m.fs.h2_tank.outlet_to_turbine.flow_mol[0].fix(turb_flow)
    m.fs.h2_tank.outlet_to_pipeline.flow_mol[0].fix(0)
    m.fs.h2_tank.properties_in[0].flow_mol.fix()
    m.fs.h2_tank.properties_in[0].mole_frac_comp.fix()
    m.fs.h2_tank.properties_in[0].component('temperature').fix()
    m.fs.h2_tank.properties_in[0].component('pressure').fix()
    m.fs.h2_tank.initialize(outlvl=outlvl)
    if turb_energy_max < 1:
        m.fs.h2_tank.outlet_to_turbine.flow_mol[0].unfix()


def heuristic_follow_dispatch(tracker, params, dispatch, profiles, target_profiles, verbose=False):
    active_blks = tracker.model.fs.windBatteryHydrogen.get_active_process_blocks()

    wind_gen_max = [pyo.value(b.fs.windpower.system_capacity * b.fs.windpower.capacity_factor[0]) for b in active_blks]
    wind_diff = wind_gen_max - dispatch * 1e3
    wind_excess = np.clip(wind_diff, 0, None)
    wind_shortage = np.clip(-wind_diff, 0, None)

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

        if wind_excess[i] > 0:

            energy_to_battery_ts = min(min(batt_energy_max, batt_soc_max - batt_soc), wind_excess[i])
            if energy_to_battery_ts < 1:
                energy_to_battery_ts = 0
            if energy_to_battery_ts / 0.95 > wind_excess[i]:
                energy_to_battery_ts = wind_excess[i] * 0.95
            wind_excess[i] -= energy_to_battery_ts / 0.95
            batt_soc += energy_to_battery_ts
            
            energy_to_pem_ts = min(min(params['pem_mw'] * 1e3, tank_energy_charge_max - tank_energy_charged), wind_excess[i])
            if energy_to_pem_ts < 1:
                energy_to_pem_ts = 0
            tank_energy_charged += energy_to_battery_ts

            batt_out = 0
            turb_out = 0
            wind_energy -= wind_excess[i]

        elif wind_shortage[i] > 0:
            turb_out = min(min(tank_energy_discharge_max, turb_energy_max), wind_shortage[i])
            batt_out = min(min(batt_soc, batt_energy_max), wind_shortage[i])

            if params['discharge_first'] == "tank":
                tank_energy_discharge_max -= turb_out
                if tank_energy_discharge_max < 1:
                    turb_out = max(turb_out - 1, 0)
                wind_shortage[i] -= turb_out
                batt_out = min(batt_out, wind_shortage[i])
                if batt_out / 0.95 > batt_soc:
                    batt_out = batt_soc * 0.95
                batt_soc -= batt_out / 0.95
                wind_shortage[i] -= batt_out
            else:
                if batt_out / 0.95 > batt_soc:
                    batt_out = batt_soc * 0.95
                batt_soc -= batt_out / 0.95
                wind_shortage[i] -= batt_out
                turb_out = min(turb_out, wind_shortage[i])
                tank_energy_discharge_max -= turb_out
                if tank_energy_discharge_max < 1:
                    turb_out = max(turb_out - 1, 0)
                wind_shortage[i] -= turb_out
            energy_to_battery_ts = 0
            energy_to_pem_ts = 0
            if abs(wind_shortage[i] > 1e-3):
                print("Missed Dispatched: ", wind_shortage[i])
                # raise Exception
        else:
            batt_out = energy_to_battery_ts = energy_to_pem_ts = 0

        init_with_fixed_controls(process_blk, wind_energy, energy_to_battery_ts, energy_to_pem_ts, batt_soc, batt_out, turb_out, turb_energy_max, params['turb_conv'], verbose)

    print(energy_to_battery_ts, batt_out, energy_to_pem_ts)    


def init_dtree_model(params):
    year_results = params['year_results']
    year_results['Wind Excess [MW]'] = (year_results['Wind Max [MW]'] - year_results['Load [MW]']).clip(lower=0)
    year_results['Wind Shortage [MW]'] = (year_results['Load [MW]'] - year_results['Wind Max [MW]']).clip(lower=0)
    year_results['Previous State of Charge [MWh]'] = np.roll(year_results['State of Charge [MWh]'], 1)
    year_results['Previous Tank Holdup [kg]'] = np.roll(year_results['Tank Holdup [kg]'], 1)
    year_results['Ratio Battery Input to Nameplate [MW]'] = year_results['Wind Power to Battery [MW]'] / params['batt_mw']
    year_results['Ratio PEM Input to Nameplate [MW]'] = year_results['Wind Power to PEM [MW]'] / params['pem_mw']
    year_results['Ratio Battery Output to Nameplate [MW]'] = year_results['Battery Power Output [MW]'] / params['batt_mw']
    year_results['Ratio Turbine Output to Nameplate [MW]'] = year_results['Turbine Power Output [MW]'] / params['turb_mw']

    year_results['4-Hr Ahead Wind Excess [MW]'] = year_results['Wind Excess [MW]'][::-1].rolling(4, min_periods=0).sum()
    year_results['16-Hr Ahead Wind Excess [MW]'] = year_results['Wind Excess [MW]'][::-1].rolling(16, min_periods=0).sum()
    year_results['24-Hr Ahead Wind Excess [MW]'] = year_results['Wind Excess [MW]'][::-1].rolling(24, min_periods=0).sum()

    year_results['4-Hr Ahead Wind Shortage [MW]'] = year_results['Wind Shortage [MW]'][::-1].rolling(4, min_periods=0).sum()
    year_results['16-Hr Ahead Wind Shortage [MW]'] = year_results['Wind Shortage [MW]'][::-1].rolling(16, min_periods=0).sum()
    year_results['24-Hr Ahead Wind Shortage [MW]'] = year_results['Wind Shortage [MW]'][::-1].rolling(24, min_periods=0).sum()

    year_results.columns

    input_cols = ['Previous State of Charge [MWh]', 'Previous Tank Holdup [kg]',
        'Wind Excess [MW]', 'Wind Shortage [MW]',  
        '4-Hr Ahead Wind Excess [MW]', 
        '16-Hr Ahead Wind Excess [MW]', '24-Hr Ahead Wind Excess [MW]',
        '4-Hr Ahead Wind Shortage [MW]', 
        '16-Hr Ahead Wind Shortage [MW]', '24-Hr Ahead Wind Shortage [MW]']
    output_cols = ['Ratio Battery Input to Nameplate [MW]',
        'Ratio PEM Input to Nameplate [MW]',
        'Ratio Battery Output to Nameplate [MW]',
        'Ratio Turbine Output to Nameplate [MW]']
    X = year_results[input_cols].to_numpy()
    y = year_results[output_cols].to_numpy()

    dtree_model = DecisionTreeRegressor(random_state=1, max_depth=12)
    dtree_model.fit(X, y)
    print(dtree_model.score(X, y))
    return dtree_model


dtree_model = None
def dtree(tracker, params, dispatch, profiles, target_profiles, verbose=False):
    global dtree_model
    if dtree_model is None:
        dtree_model = init_dtree_model(params)

    active_blks = tracker.model.fs.windBatteryHydrogen.get_active_process_blocks()

    timestep = target_profiles['timestep']
    wind_gen_max = np.array(params['year_results']["Wind Max [MW]"][timestep:timestep+24+len(active_blks)-1])
    dispatch = np.array(params['load'][timestep:timestep+24+len(active_blks)-1])
    wind_diff = wind_gen_max - dispatch
    wind_excess = np.clip(wind_diff, 0, None)
    wind_shortage = np.clip(-wind_diff, 0, None)

    batt_soc = profiles['realized_soc'][0] * 1e-3
    tank_soc = profiles['realized_h2_tank_holdup'][0] / h2_mols_per_kg

    turb_energy_max = params['turb_mw'] * 1e3

    for i, process_blk in enumerate(active_blks):
        wind_excess_4hr = sum(wind_excess[i:i+4])
        wind_excess_16hr = sum(wind_excess[i:i+16])
        wind_excess_24hr = sum(wind_excess[i:i+24])
        wind_shortage_4hr = sum(wind_shortage[i:i+4])
        wind_shortage_16hr = sum(wind_shortage[i:i+16])
        wind_shortage_24hr = sum(wind_shortage[i:i+24])

        X = np.array((batt_soc, tank_soc, wind_excess[0], wind_shortage[0], wind_excess_4hr, wind_excess_16hr, wind_excess_24hr,
            wind_shortage_4hr, wind_shortage_16hr, wind_shortage_24hr))
        y = dtree_model.predict(X.reshape(1, -1))[0]

        energy_to_battery_ts = max(0, min(y[0], 1)) * params['batt_mw']
        energy_to_pem_ts = max(0, min(y[1], 1)) * params['pem_mw']
        batt_out = max(0, min(y[2], 1)) * params['batt_mw']
        turb_out = max(0, min(y[3], 1)) * params['turb_mw']

        wind_energy = dispatch[i] + energy_to_battery_ts + energy_to_pem_ts - batt_out - turb_out
        wind_energy = max(0, min(wind_energy, wind_gen_max[i]))

        init_with_fixed_controls(process_blk, wind_energy, energy_to_battery_ts, energy_to_pem_ts, batt_soc, batt_out, turb_out, turb_energy_max, params['turb_conv'], verbose)
        pass
