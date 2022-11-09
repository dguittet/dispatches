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
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo
import idaes.logger as idaeslog
from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds, log_close_to_bounds
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
from dispatches.case_studies.renewables_case.RE_flowsheet import create_model, propagate_state, value, h2_mols_per_kg, PA, battery_ramp_rate
from dispatches.case_studies.renewables_h2_case.re_h2_parameters import re_h2_parameters, kg_to_tons, n_hrs, re_h2_dir


def wind_battery_hydrogen_variable_pairs(m1, m2):
    """
    This function links together unit model state variables from one timestep to the next.

    The simple hydrogen tank and the battery model have material and energy holdups that need to be consistent across time blocks.

    Args:
        m1: current time block model
        m2: next time block model
    """
    pairs = [(m1.fs.h2_tank.tank_holdup[0], m2.fs.h2_tank.tank_holdup_previous[0]),
             (m1.fs.h2_tank.tank_throughput[0], m2.fs.h2_tank.tank_throughput_previous[0])]
    pairs += [(m1.fs.battery.state_of_charge[0], m2.fs.battery.initial_state_of_charge),
              (m1.fs.battery.energy_throughput[0], m2.fs.battery.initial_energy_throughput),
              (m1.fs.battery.nameplate_power, m2.fs.battery.nameplate_power)]
    
    return pairs


def wind_battery_hydrogen_periodic_variable_pairs(m1, m2):
    """
    The final hydrogen material holdup and battery storage of charge must be the same as in the intial timestep. 

    Args:
        m1: final time block model
        m2: first time block model
    """
    pairs = [(m1.fs.h2_tank.tank_holdup[0], m2.fs.h2_tank.tank_holdup_previous[0])]
    pairs += [(m1.fs.battery.state_of_charge[0], m2.fs.battery.initial_state_of_charge),
              (m1.fs.battery.nameplate_power, m2.fs.battery.nameplate_power)]
    return pairs


def initialize_fs(m, input_params=dict(), verbose=False):
    """
    Initializing the flowsheet is done starting with the wind model and propagating the solved initial state to downstream models.

    The splitter is initialized with no flow to the battery or PEM so all electricity flows to the grid, which makes the initialization of all
    unit models downstream of the wind plant independent of its time-varying electricity production. This initialzation function can
    then be repeated for all timesteps within a dynamic analysis.

    Args:
        m: model
        verbose:
    """
    outlvl = idaeslog.INFO if verbose else idaeslog.WARNING

    m.fs.windpower.initialize(outlvl=outlvl)

    propagate_state(m.fs.wind_to_splitter)
    m.fs.splitter.battery_elec[0].fix(0)
    m.fs.splitter.pem_elec[0].fix(0)
    m.fs.splitter.initialize()
    m.fs.splitter.battery_elec[0].unfix()
    m.fs.splitter.pem_elec[0].unfix()
    if verbose:
        m.fs.splitter.report(dof=True)

    propagate_state(m.fs.splitter_to_pem)
    propagate_state(m.fs.splitter_to_battery)

    batt_init = input_params['batt_mw'] * 1e3
    if 'batt_soc_init_mwh' in input_params.keys():
        batt_init = input_params['batt_soc_init_mwh'] * 1e3
    m.fs.battery.initial_state_of_charge.fix(batt_init)
    m.fs.battery.initial_energy_throughput.fix(batt_init)
    m.fs.battery.elec_in[0].fix()
    m.fs.battery.elec_out[0].fix(value(m.fs.battery.elec_in[0]))
    m.fs.battery.initialize(outlvl=outlvl)
    m.fs.battery.elec_in[0].unfix()
    m.fs.battery.elec_out[0].unfix()
    m.fs.battery.initial_state_of_charge.unfix()
    m.fs.battery.initial_energy_throughput.unfix()
    if verbose:
        m.fs.battery.report(dof=True)

    m.fs.pem.initialize(outlvl=outlvl)
    if verbose:
        m.fs.pem.report(dof=True)

    propagate_state(m.fs.pem_to_tank)

    tank_throughput_init = 0
    tank_holdup_init = input_params['turb_mw'] * 1e3 / input_params['turb_conv'] * h2_mols_per_kg
    if 'tank_holdup_init_mols' in input_params.keys():
        tank_holdup_init = input_params['tank_holdup_init_mols']
    m.fs.h2_tank.outlet_to_turbine.flow_mol[0].fix(value(m.fs.h2_tank.inlet.flow_mol[0]))
    m.fs.h2_tank.outlet_to_pipeline.flow_mol[0].fix(0)
    m.fs.h2_tank.tank_holdup_previous.fix(tank_holdup_init)
    m.fs.h2_tank.tank_throughput_previous.fix(tank_throughput_init)
    m.fs.h2_tank.initialize(outlvl=outlvl)
    m.fs.h2_tank.outlet_to_turbine.flow_mol[0].unfix()
    m.fs.h2_tank.outlet_to_pipeline.flow_mol[0].unfix()
    m.fs.h2_tank.tank_holdup_previous.unfix()
    m.fs.h2_tank.tank_throughput_previous.unfix()


def wind_battery_hydrogen_model(wind_resource_config, input_params, verbose):
    """
    Creates an initialized flowsheet model for a single time step with operating, size and cost components
    
    First, the model is created using the input_params and wind_resource_config
    Second, the model is initialized so that it solves and its values are internally consistent
    Third, battery ramp constraints and operating cost components are added

    Args:
        wind_resource_config: wind resource for the time step
        input_params: size and operation parameters. Required keys: `wind_mw`, `pem_bar`, `batt_mw`, `tank_size`, `pem_bar`, `turb_conv`
        verbose:
    """
    m = create_model(input_params['wind_mw'], input_params['pem_bar'], input_params['batt_mw'], "simple", input_params['tank_size'], None,
                     wind_resource_config)
    if 'batt_hr' in input_params.keys():
        input_params['batt_mwh'] = input_params['batt_mw'] * input_params['batt_hr']
    if 'batt_mwh' in input_params.keys():
        m.fs.battery.nameplate_energy.fix(input_params['batt_mwh'] * 1e3)

    m.fs.h2_turbine_elec = pyo.Expression(expr=m.fs.h2_tank.outlet_to_turbine.flow_mol[0] * 3600 / h2_mols_per_kg * input_params['turb_conv'])

    initialize_fs(m, input_params, verbose=verbose)

    # unfix for design optimization
    if input_params['design_opt']:
        m.fs.battery.nameplate_power.unfix()
        m.fs.battery.nameplate_energy.unfix()
        m.fs.windpower.system_capacity.unfix()
    else:
        m.fs.windpower.system_capacity.fix(input_params['wind_mw'] * 1e3)
        m.fs.battery.nameplate_power.fix(input_params['batt_mw'] * 1e3)
        m.fs.battery.nameplate_energy.fix(input_params['batt_mwh'] * 1e3)

    batt = m.fs.battery

    batt.degradation_rate.set_value(0)

    batt.energy_down_ramp = pyo.Constraint(
        expr=batt.initial_state_of_charge - batt.state_of_charge[0] <= battery_ramp_rate)
    batt.energy_up_ramp = pyo.Constraint(
        expr=batt.state_of_charge[0] - batt.initial_state_of_charge <= battery_ramp_rate)

    return m


def wind_battery_hydrogen_mp_block(wind_resource_config, input_params, verbose):
    """
    Wrapper of `wind_battery_hydrogen_model` for creating the process model per time point for the MultiPeriod model.
    Uses cloning of the Pyomo model in order to reduce runtime. 
    The clone is reinitialized with the `wind_resource_config` for the given time point, which only required modifying
    the windpower and the splitter, as the rest of the units have no flow and therefore is unaffected by wind resource changes.

    Args:
        wind_resource_config: dictionary with `resource_speed` for the time step
        input_params: size and operation parameters. Required keys: `wind_mw`, `pem_bar`, `batt_mw`, `tank_size`, `pem_bar`, `turb_conv`
        verbose:
    """

    if 'pyo_model' not in input_params.keys():
        input_params['pyo_model'] = wind_battery_hydrogen_model(wind_resource_config, input_params, verbose)
    m = input_params['pyo_model'].clone()

    if 'resource_speed' in wind_resource_config.keys():
        m.fs.windpower.config.resource_speed = wind_resource_config['resource_speed']
    elif 'capacity_factor' in wind_resource_config.keys():
        m.fs.windpower.config.capacity_factor = wind_resource_config['capacity_factor']
    else:
        raise ValueError(f"`wind_resource_config` dict must contain either 'resource_speed' or 'capacity_factor' values")

    m.fs.windpower.setup_resource()

    outlvl = idaeslog.INFO if verbose else idaeslog.WARNING
    m.fs.windpower.initialize(outlvl=outlvl)
    propagate_state(m.fs.wind_to_splitter)
    m.fs.splitter.initialize()
    return m


def size_constraints(mp_model, input_params):
    m = mp_model.pyomo_model
    blks = mp_model.get_active_process_blocks()

    m.wind_system_capacity = pyo.Param(default=input_params['wind_mw'] * 1e3, units=pyo.units.kW)
    m.wind_add_system_capacity = pyo.Var(domain=pyo.NonNegativeReals, initialize=0, units=pyo.units.kW)
    if not input_params['build_add_wind']:
        m.wind_add_system_capacity.fix(0)
    m.battery_system_capacity = pyo.Var(domain=pyo.NonNegativeReals, initialize=input_params['batt_mw'] * 1e3, units=pyo.units.kW)
    m.battery_system_energy = pyo.Var(domain=pyo.NonNegativeReals, initialize=(input_params['batt_mwh'] if 'batt_mwh' in input_params.keys()
                                                                                else (input_params['batt_mw'] * input_params['batt_hr'])) * 1e3, units=pyo.units.kWh)
    m.pem_system_capacity = pyo.Var(domain=pyo.NonNegativeReals, initialize=input_params['pem_mw'] * 1e3, units=pyo.units.kW)
    m.h2_tank_size = pyo.Var(domain=pyo.NonNegativeReals, initialize=input_params['tank_size'], units=pyo.units.kg)
    m.turb_system_capacity = pyo.Var(domain=pyo.NonNegativeReals, initialize=input_params['turb_mw'] * 1e3, units=pyo.units.kW)

    if not input_params['design_opt']:
        m.battery_system_capacity.fix()
        m.battery_system_energy.fix()
        m.pem_system_capacity.fix()
        m.h2_tank_size.fix()
        m.turb_system_capacity.fix()

    m.wind_max_p = pyo.Constraint(mp_model.pyomo_model.TIME, rule=lambda b, t: blks[t].fs.windpower.system_capacity <= m.wind_system_capacity + m.wind_add_system_capacity)
    m.battery_max_p = pyo.Constraint(mp_model.pyomo_model.TIME, rule=lambda b, t: blks[t].fs.battery.nameplate_power <= m.battery_system_capacity)
    m.battery_max_e = pyo.Constraint(mp_model.pyomo_model.TIME, rule=lambda b, t: blks[t].fs.battery.nameplate_energy <= m.battery_system_energy)
    m.pem_max_p = pyo.Constraint(mp_model.pyomo_model.TIME, rule=lambda b, t: blks[t].fs.pem.electricity[0] <= m.pem_system_capacity)
    m.tank_max_p = pyo.Constraint(mp_model.pyomo_model.TIME, rule=lambda b, t: blks[t].fs.h2_tank.tank_holdup[0] / h2_mols_per_kg <= m.h2_tank_size)
    m.turb_max_p = pyo.Constraint(mp_model.pyomo_model.TIME, rule=lambda b, t: blks[t].fs.h2_turbine_elec <= m.turb_system_capacity)

def calculate_capital_costs(m, input_params):
    # capital costs
    m.wind_cap_cost = pyo.Param(default=input_params["wind_cap_cost"], mutable=True)
    m.pem_cap_cost = pyo.Param(default=input_params["pem_cap_cost"], mutable=True)
    m.batt_cap_cost_kw = pyo.Param(default=input_params["batt_cap_cost_kw"], mutable=True)
    m.batt_cap_cost_kwh = pyo.Param(default=input_params["batt_cap_cost_kwh"], mutable=True)
    m.tank_cap_cost = pyo.Param(default=input_params["tank_cap_cost_per_kg"], mutable=True)
    m.turb_cap_cost = pyo.Param(default=input_params["turbine_cap_cost"], mutable=True)

    m.total_cap_cost = pyo.Expression(expr=m.wind_cap_cost * m.wind_add_system_capacity
                                       + m.batt_cap_cost_kw * m.battery_system_capacity
                                       + m.batt_cap_cost_kwh * m.battery_system_energy
                                       + m.pem_cap_cost * m.pem_system_capacity
                                       + m.tank_cap_cost * m.h2_tank_size
                                       + m.turb_cap_cost * m.turb_system_capacity)


def calculate_fixed_costs(m, input_params):
    m.windpower_op_cost_unit = pyo.Param(
        initialize=input_params["wind_op_cost"],
        doc="fixed cost of operating wind plant $/kW-yr")
    m.pem_op_cost_unit = pyo.Param(
        initialize=input_params["pem_op_cost"],
        doc="fixed cost of operating pem $/kW-yr")
    m.h2_tank_op_cost_unit = pyo.Param(
        initialize=input_params["tank_op_cost"],
        doc="fixed cost of operating tank in $/kg-yr")
    m.h2_turbine_op_cost_unit = pyo.Param(
        initialize=input_params["turbine_op_cost"],
        doc="fixed cost of operating turbine in $/kW-yr")

    m.annual_fixed_cost = pyo.Expression(expr=m.wind_system_capacity * m.windpower_op_cost_unit
                                              + m.pem_system_capacity * m.pem_op_cost_unit
                                              + m.h2_tank_size * m.h2_tank_op_cost_unit
                                              + m.turb_system_capacity * m.h2_turbine_op_cost_unit)


def calculate_variable_costs(mp_model, input_params):
    m.battery_var_cost_unit = pyo.Param(
        initialize=input_params["batt_rep_cost_kwh"],
        doc="variable cost of battery degradation $/kwH")
    m.pem_var_cost_unit = pyo.Param(
        initialize=input_params["pem_var_cost"],
        doc="variable operating cost of pem $/kWh")
    m.h2_turbine_var_cost_unit = pyo.Param(
        initialize=input_params["turbine_var_cost"],
        doc="variable cost of operating turbine in $/kWh")

    m = mp_model.pyomo_model
    blks = mp_model.get_active_process_blocks()

    for blk in blks:
        blk_battery = blk.fs.battery
        blk_pem = blk.fs.pem
        blk_turb = blk.fs.h2_turbine

        blk_battery.var_cost = pyo.Expression(
            expr=blk_battery.degradation_rate * (blk_battery.energy_throughput[0] - blk_battery.initial_energy_throughput) * m.battery_var_cost_unit)
        blk_pem.var_cost = pyo.Expression(
            expr=m.pem_var_cost_unit * blk_pem.electricity[0])
        blk_turb.var_cost = pyo.Expression(
            expr=m.h2_turbine_var_cost_unit * blk.fs.h2_turbine_elec
        )
        blk.var_total_cost = pyo.Expression(expr=blk_pem.var_cost
                                                 + blk_battery.var_cost
                                                 + blk_turb.var_cost)


def add_load_following_obj(mp_model, input_params):
    m = mp_model.pyomo_model
    blks = mp_model.get_active_process_blocks()
    n_weeks = len(blks) / (7 * 24)

    wind_mw = [input_params['wind_mw'] * i[1]['wind_resource_config']['capacity_factor'][0] for i in input_params['wind_resource'].items()]

    for (i, blk) in enumerate(blks):
        blk_battery = blk.fs.battery
        blk_tank = blk.fs.h2_tank

        blk.load_power = pyo.Param(default=input_params['load'][i] * 1e3, mutable=True, units=pyo.units.kW)   # convert to kW
        blk.output_power = pyo.Expression(expr=blk.fs.splitter.grid_elec[0] + blk_battery.elec_out[0] + blk.fs.h2_turbine_elec)
        blk.under_power = pyo.Var(domain=pyo.NonNegativeReals, initialize=0, units=pyo.units.kW)
        blk.over_power = pyo.Var(domain=pyo.NonNegativeReals, initialize=0, units=pyo.units.kW)

        blk.meet_load = pyo.Constraint(expr=blk.output_power + blk.under_power == blk.load_power + blk.over_power)

        blk.costs = pyo.Expression(expr=input_params['shortfall_price'] * blk.under_power + blk.var_total_cost)
        blk.hydrogen_revenue = pyo.Expression(expr=m.h2_price_per_kg / h2_mols_per_kg * blk_tank.outlet_to_pipeline.flow_mol[0] * 3600)

        if 'modop' in input_params.keys() and input_params['modop']:
            if wind_mw[i] > input_params['load'][i]:
                blk.no_batt_discharge = pyo.Constraint(expr=blk_battery.elec_out[0] == 0)
                blk.no_tank_discharge = pyo.Constraint(expr=blk.fs.h2_turbine_elec == 0)
            else:
                blk.no_batt_charge = pyo.Constraint(expr=blk_battery.elec_in[0] == 0)
                blk.no_tank_charge = pyo.Constraint(expr=blk.fs.pem.electricity[0] == 0)

        if 'min_batt_soc' in input_params.keys():
            blk.min_batt_soc = pyo.Constraint(expr=blk_battery.state_of_charge[0] >= input_params['min_batt_soc'] * 1e3)

        if 'min_tank_soc' in input_params.keys():
            blk.min_tank_soc = pyo.Constraint(expr=blk_tank.tank_holdup[0] / h2_mols_per_kg * input_params['turb_conv'] >= input_params['min_tank_soc'] * 1e3)

    m.annual_revenue = pyo.Expression(expr=(sum([-blk.costs + blk.hydrogen_revenue for blk in blks])) * 52.143 / n_weeks
                                           - m.annual_fixed_cost)

    m.NPV = pyo.Expression(expr=-m.total_cap_cost + PA * m.annual_revenue)
    m.obj = pyo.Objective(expr=-m.NPV * 1e-8)


def add_surrogate_obj(mp_model, input_params):
    # surrogate model parameters
    L_a, L_b, L_c, L_d, L_e = (1.02629254, 1.01123214, 17.05785483, 0.58490096, 0.05185373)
    k_a, k_b, k_c = (3.74116613, -5.37471860e-02, 2.85194646e-03)
    A_0, A_a, A_b = (31.47032532, -36.14239165, 4.41310189)
    w_0, w_a, w_b = (18.55789012, 3.40416608, -12.69136832)
    m_0, m_a, m_b = (10.92758574, 0.25000692, -6.88856666)
    y_0_0, y_0_a, y_0_b = (8.56406822, -11.43706554, 0.61095527)
    max_PMaxMW = 355
    max_HR_avg = 24763
    max_hrs = 8784

    m = mp_model.pyomo_model
    blks = mp_model.get_active_process_blocks()
    n_weeks = len(blks) / (7 * 24)

    wind_mw = [input_params['wind_mw'] * i[1]['wind_resource_config']['capacity_factor'][0] for i in input_params['wind_resource'].items()]

    m.PMaxMW = pyo.Expression(expr=(m.battery_system_capacity + m.turb_system_capacity) * 1e-3)
    m.PMaxMW_ub = pyo.Constraint(expr=m.PMaxMW <= max_PMaxMW)
    m.HR_avg = pyo.Var(domain=pyo.NonNegativeReals, initialize=19119)       # Model fit from NG plants with 3.88722 $/MMBTU 
    m.HR_avg.setlb(0)
    m.HR_avg.setub(max_HR_avg)
    # m.HR_incr_1 = pyo.Expression(expr=0.617 * m.HR_avg_0)
    # m.HR_incr_2 = pyo.Expression(expr=0.684 * m.HR_avg_0)

    m.L = pyo.Expression(expr=L_a - (L_b / (1 + pyo.exp(-L_c * (m.HR_avg / max_HR_avg - L_d))) + (L_e * m.PMaxMW / max_PMaxMW)))

    m.k = pyo.Expression(expr=pyo.exp(k_a * m.HR_avg / max_HR_avg + k_b * m.PMaxMW / max_PMaxMW - k_c))
    m.x_0 = pyo.Expression(expr=0.5)

    m.L_lb = pyo.Constraint(expr=m.L >= 0)
    m.L_ub = pyo.Constraint(expr=m.L <= 1)
    m.k_lb = pyo.Constraint(expr=m.k >= 0)

    m.A = pyo.Expression(expr=A_0 + A_a * m.PMaxMW / max_PMaxMW + A_b * m.HR_avg / max_HR_avg)
    m.w = pyo.Expression(expr=w_0 + w_a * m.PMaxMW / max_PMaxMW + w_b * m.HR_avg / max_HR_avg)
    m.m = pyo.Expression(expr=m_0 + m_a * m.PMaxMW / max_PMaxMW + m_b * m.HR_avg / max_HR_avg)
    m.y_0 = pyo.Expression(expr=y_0_0 + y_0_a * m.PMaxMW / max_PMaxMW + y_0_b * m.HR_avg / max_HR_avg)

    # monthly_revenue_avg = [30.43640391, 29.05218992, 34.99922776, 30.88766605, 31.19209689,
    #    40.32723397, 54.14020062, 53.17627986, 47.71013634, 36.36529442, 30.94663306, 36.35684594]
    
    ts_per_month = len(wind_mw) // 12
    n_months = max(len(blks) // ts_per_month, 1)
    blks_month = []
    timestep, prev_timestep = 0, 0

    # convert everything to MW
    for (i, blk) in enumerate(blks):
        blk_battery = blk.fs.battery
        blk_tank = blk.fs.h2_tank

        # Make sure system meets original wind load
        blk.wind_load_power = pyo.Param(default=input_params['wind_load'][i], mutable=True, units=pyo.units.MW)   # convert to kW
        blk.output_power = pyo.Expression(expr=(blk.fs.splitter.grid_elec[0] + blk_battery.elec_out[0] + blk.fs.h2_turbine_elec) * 1e-3)
        blk.under_power = pyo.Var(domain=pyo.NonNegativeReals, initialize=0, units=pyo.units.MW)        
        blk.peaker_power = pyo.Var(domain=pyo.NonNegativeReals, initialize=0, units=pyo.units.MW)

        blk.wind_vs_peaker_power = pyo.Constraint(expr=blk.output_power + blk.under_power == blk.wind_load_power + blk.peaker_power)

        # if abs(value(blk.wind_vs_peaker_power)) > 1:
        #     wind_kw = input_params['wind_load'][i] * 1e3
        #     blk.fs.windpower.electricity[0].set_value(wind_kw)
        #     blk.fs.splitter.grid_elec[0].set_value(wind_kw)
        #     blk.fs.splitter.electricity[0].set_value(wind_kw)

        if i == 0:
            blk.peaker_cumulated_mwh = pyo.Expression(expr=blk.peaker_power)
        else:
            blk.peaker_cumulated_mwh = pyo.Expression(expr=blk.peaker_power + blks[i-1].peaker_power)

        blk.costs = pyo.Expression(expr=input_params['shortfall_price'] * blk.under_power + blk.var_total_cost)
        blk.hydrogen_revenue = pyo.Expression(expr=m.h2_price_per_kg / h2_mols_per_kg * blk_tank.outlet_to_pipeline.flow_mol[0] * 3600)

        if 'modop' in input_params.keys() and input_params['modop']:
            if wind_mw[i] > input_params['load'][i]:
                blk.no_batt_discharge = pyo.Constraint(expr=blk_battery.elec_out[0] == 0)
                blk.no_tank_discharge = pyo.Constraint(expr=blk.fs.h2_turbine_elec == 0)
            else:
                blk.no_batt_charge = pyo.Constraint(expr=blk_battery.elec_in[0] == 0)
                blk.no_tank_charge = pyo.Constraint(expr=blk.fs.pem.electricity[0] == 0)

        blk.min_storage_soc = pyo.Constraint(expr=(blk_battery.state_of_charge[0]
                                                    + blk_tank.tank_holdup[0] / h2_mols_per_kg * input_params['turb_conv']) * 1e-3 >= m.PMaxMW)

    # The cumulative peaker capacity is the battery output, turbine output, and excess wind output
    for month in range(n_months):
        prev_timestep = timestep
        timestep = min(ts_per_month * (month + 1), len(blks)) - 1
        blk = blks[timestep]
        blk.cf_cumulative_month = pyo.Expression(expr=m.L / (1 + pyo.exp(-m.k * (month/12 - m.x_0))))
        blk.storage_cumulated_mwh = pyo.Expression(expr=(blk.fs.battery.energy_throughput[0] / 2 
                                                        + blk.fs.h2_tank.tank_throughput[0] / h2_mols_per_kg * input_params['turb_conv']) * 1e-3)
        blk.meet_peaker_CF_cumulative = pyo.Constraint(expr=blk.peaker_cumulated_mwh >= blk.cf_cumulative_month * max_hrs * m.PMaxMW)

        blk.avg_revenue_per_mwh = pyo.Expression(expr=(m.A * pyo.exp(-(m.w * month/12 - m.m)**2) + m.y_0))
        blk.cf_sold_month = pyo.Var(domain=pyo.NonNegativeReals, initialize=0)
        if month == 0:
            blk.cf_sold_month_ub = pyo.Constraint(expr=blk.cf_sold_month <= blk.cf_cumulative_month)
        else:
            blk.cf_sold_month_ub = pyo.Constraint(expr=blk.cf_sold_month <= blk.cf_cumulative_month - blks_month[-1].cf_cumulative_month)
            

        blk.revenue = pyo.Expression(expr=blk.cf_sold_month * blk.avg_revenue_per_mwh * max_hrs * m.PMaxMW)
        blks_month.append(blk)

    m.annual_revenue = pyo.Expression(expr=sum([-blk.costs + blk.hydrogen_revenue for blk in blks]) * 52.143 / n_weeks
                                            + sum([blk.revenue for blk in blks_month]) * 12 / n_months
                                            - m.annual_fixed_cost)

    m.NPV = pyo.Expression(expr=-m.total_cap_cost + PA * m.annual_revenue)
    m.obj = pyo.Objective(expr=-m.NPV * 1e-8)
    return blks_month


def wind_battery_hydrogen_optimize(n_time_points, input_params, verbose=False, plot=False):
    """
    The main function for optimizing the flowsheet's design and operating variables for Net Present Value. 

    Creates the MultiPeriodModel and adds the size and operating constraints in addition to the Net Present Value Objective.
    The NPV is a function of the capital costs, the electricity market profit, the hydrogen market profit, and the capital recovery factor.
    The operating decisions and state evolution of the unit models and the flowsheet as a whole form the constraints of the Non-linear Program.

    Required input parameters include:
        `wind_mw`: initial guess of the wind size
        `wind_mw_ub`: upper bound of wind size
        `batt_mw`: initial guess of the battery size
        `pem_mw`: initial guess of the pem size
        `pem_bar`: operating pressure
        `pem_temp`: operating temperature [K]
        `tank_size`: initial guess of the tank_size [kg H2]
        `turb_mw`: intial guess of the turbine size
        `turb_conv`: h2 conversion rate kWh/kgH2
        `wind_resource`: dictionary of wind resource configs for each time point
        `h2_price_per_kg`: market price of hydrogen
        `DA_LMPs`: LMPs for each time point
        `build_add_wind`: if false, fix wind size to initial size and do not add wind capital cost to NPV.
            otherwise, any additional wind beyond that `wind_mw` incurs capital cost 

    Args:
        n_time_points: number of periods in MultiPeriod model
        input_params: 
        verbose: print all logging and outputs from unit models, initialization, solvers, etc
        plot: plot the operating variables time series
    """
    # create the multiperiod model object
    n_weeks = n_time_points / (7 * 24)
    mp_model = MultiPeriodModel(n_time_points=n_time_points,
                                process_model_func=partial(wind_battery_hydrogen_mp_block, input_params=input_params, verbose=verbose),
                                linking_variable_func=wind_battery_hydrogen_variable_pairs,
                                periodic_variable_func=wind_battery_hydrogen_periodic_variable_pairs)

    mp_model.build_multi_period_model(input_params['wind_resource'])

    m = mp_model.pyomo_model
    blks = mp_model.get_active_process_blocks()
    # blks[0].fs.battery.initial_energy_throughput.fix()
    blks[0].fs.h2_tank.tank_throughput_previous.fix(0)
    
    size_constraints(mp_model, input_params)
    
    # Add hydrogen market
    m.h2_price_per_kg = pyo.Param(default=input_params['h2_price_per_kg'], mutable=True)

    calculate_capital_costs(m, input_params)
    calculate_fixed_costs(m, input_params)
    calculate_variable_costs(mp_model, input_params)

    solvers_list = ['xpress_direct', 'cbc', 'ipopt']
    if input_params['opt_mode'] == "meet_load":
        add_load_following_obj(mp_model, input_params)
    elif input_params['opt_mode'] == "surrogate":
        blks_month = add_surrogate_obj(mp_model, input_params)
        solvers_list = ['ipopt']

    opt = None
    for solver in solvers_list:
        if pyo.SolverFactory(solver).available(exception_flag=False):
            opt = pyo.SolverFactory(solver)
            break
    if not opt:
        raise RuntimeWarning("No available solvers")

    opt.options['tol'] = 1e-7
    opt.options['max_iter'] = 200

    if verbose:
        solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO, tag="properties")
        log_infeasible_constraints(m, logger=solve_log, tol=1e-4, log_expression=True, log_variables=True)
        log_infeasible_bounds(m, logger=solve_log, tol=1e-4)

    opt.solve(m, tee=True)

    if verbose:
        solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO, tag="properties")
        log_infeasible_constraints(m, logger=solve_log, tol=1e-4, log_expression=False, log_variables=False)
        log_infeasible_bounds(m, logger=solve_log, tol=1e-4)

    h2_prod = [pyo.value(blks[i].fs.pem.outlet_state[0].flow_mol * 3600 / h2_mols_per_kg) for i in range(n_time_points)]
    h2_tank_in = [pyo.value(blks[i].fs.h2_tank.inlet.flow_mol[0] * 3600 / h2_mols_per_kg) for i in range(n_time_points)]
    h2_tank_out = [pyo.value((blks[i].fs.h2_tank.outlet_to_pipeline.flow_mol[0] + blks[i].fs.h2_tank.outlet_to_turbine.flow_mol[0]) * 3600 / h2_mols_per_kg) for i in range(n_time_points)]
    h2_tank_holdup = [pyo.value(blks[i].fs.h2_tank.tank_holdup[0]) / h2_mols_per_kg for i in range(n_time_points)]
    h2_sales = [pyo.value(blks[i].fs.h2_tank.outlet_to_pipeline.flow_mol[0] * 3600 / h2_mols_per_kg) for i in range(n_time_points)]
    h2_turbine_in = [pyo.value(blks[i].fs.h2_tank.outlet_to_turbine.flow_mol[0] * 3600 / h2_mols_per_kg) for i in range(n_time_points)]

    wind_gen = [pyo.value(blks[i].fs.windpower.electricity[0]) for i in range(n_time_points)]
    wind_out = [pyo.value(blks[i].fs.splitter.grid_elec[0]) for i in range(n_time_points)]
    wind_to_pem = [pyo.value(blks[i].fs.pem.electricity[0]) for i in range(n_time_points)]
    batt_out = [pyo.value(blks[i].fs.battery.elec_out[0]) for i in range(n_time_points)]
    batt_in = [pyo.value(blks[i].fs.battery.elec_in[0]) for i in range(n_time_points)]
    batt_soc = [pyo.value(blks[i].fs.battery.state_of_charge[0]) for i in range(n_time_points)]
    h2_turbine_elec = [pyo.value(blks[i].fs.h2_turbine_elec) for i in range(n_time_points)]
    
    elec_costs = [pyo.value(blks[i].costs) for i in range(n_time_points)]
    h2_revenue = [pyo.value(blks[i].hydrogen_revenue) for i in range(n_time_points)]
    under_power = [pyo.value(blks[i].under_power) for i in range(n_time_points)]

    hours = np.arange(n_time_points)

    wind_cap = value(m.wind_system_capacity + m.wind_add_system_capacity) * 1e-3
    batt_cap = value(m.battery_system_capacity) * 1e-3
    batt_energy = value(m.battery_system_energy) * 1e-3

    pem_cap = value(m.pem_system_capacity) * 1e-3
    tank_size = value(m.h2_tank_size) * kg_to_tons # to ton
    turb_cap = value(m.turb_system_capacity) * 1e-3

    design_res = {
        'wind_mw': wind_cap,
        "batt_mw": batt_cap,
        "batt_mwh": batt_energy,
        "pem_mw": pem_cap,
        "tank_tonH2": tank_size,
        "turb_mw": turb_cap,
        "annual_under_power": sum(under_power) * 52/ n_weeks,
        "annual_rev_h2": sum(h2_revenue) * 52 / n_weeks,
        "annual_costs_E": sum(elec_costs) * 52 / n_weeks,
        "NPV": value(m.NPV),
        "capital_cost": value(m.total_cap_cost)
    }

    if input_params['opt_mode'] == 'surrogate':
        peaker_power = [pyo.value(blks[i].peaker_power) for i in range(n_time_points)]
        peaker_cumulated_mwh = [pyo.value(blks[i].peaker_cumulated_mwh) for i in range(n_time_points)]

        revenue = [pyo.value(blk.revenue) for blk in blks_month]
        cf_sold_month = [pyo.value(blk.cf_sold_month) for blk in blks_month]
        storage_cumulated_mwh = [pyo.value(blk.storage_cumulated_mwh) for blk in blks_month]
        peaker_cumulated_mwh = [pyo.value(blk.peaker_cumulated_mwh) for blk in blks_month]
        avg_revenue_per_mwh = [pyo.value(blk.avg_revenue_per_mwh) for blk in blks_month]
        cf_cumulative_month = [pyo.value(blk.cf_cumulative_month) for blk in blks_month]

        design_res["annual_rev_E"] = sum(revenue) * 8784 / n_time_points
        design_res["annual_sold_E"] = sum(cf_sold_month) * 8784 / n_time_points * value(m.PMaxMW)
        design_res["avg_revenue_per_mwh"] = sum(avg_revenue_per_mwh) / len(avg_revenue_per_mwh)
        design_res['PMaxMW'] = value(m.PMaxMW)
        design_res['Bid_$/MW'] = value(m.HR_avg) * 3.88722 * 1e-3 # Model fit from NG plants with 3.88722 $/MMBTU 
        design_res['Annual Peaker CF'] = cf_cumulative_month[-1]

    print(design_res)

    if plot:
        fig, axs = plt.subplots(3, 1, figsize=(20, 8))
        fig.suptitle(f"Optimal NPV ${round(value(m.NPV) * 1e-6)}mil from {round(batt_cap, 2)} MW Battery, "
                     f"{round(pem_cap, 2)} MW PEM, {round(tank_size, 2)} tonH2 Tank and {round(turb_cap, 2)} MW Turbine")

        # color = 'tab:green'
        axs[0].set_xlabel('Hour')
        axs[0].set_ylabel('kW')
        axs[0].step(hours, wind_gen, label="Wind Generation [kW]")
        axs[0].step(hours, wind_out, label="Wind to Grid [kW]")
        axs[0].step(hours, wind_to_pem, label="Wind to Pem [kW]")
        axs[0].step(hours, batt_in, label="Wind to Batt [kW]")
        axs[0].step(hours, batt_out, label="Batt to Grid [kW]")
        axs[0].step(hours, h2_turbine_elec, label="H2 Turbine [kW]")
        axs[0].tick_params(axis='y', )
        axs[0].legend()
        axs[0].grid(visible=True, which='major', color='k', linestyle='--', alpha=0.2)
        axs[0].minorticks_on()
        axs[0].grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.2)

        # ax1[1].set_ylabel('kg/hr', )
        axs[1].step(hours, h2_prod, label="PEM H2 production [kg/hr]")
        axs[1].step(hours, h2_tank_in, label="Tank inlet [kg/hr]")
        axs[1].step(hours, h2_tank_out, label="Tank outlet [kg/hr]")
        axs[1].step(hours, h2_tank_holdup, label="Tank holdup [kg]")
        axs[1].tick_params(axis='y', )
        axs[1].legend()
        axs[1].grid(visible=True, which='major', color='k', linestyle='--', alpha=0.2)
        axs[1].minorticks_on()
        axs[1].grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.2)

        if input_params['opt_mode'] != 'surrogate':
            # plot costs
            axs[2].step(hours, elec_costs, label="Elec Cost [$]")
            axs[2].step(hours, h2_revenue, label="H2 Rev [$]")
            axs[2].step(hours, np.cumsum(elec_costs), label="Elec Cost cumulative [$]")
            axs[2].step(hours, np.cumsum(h2_revenue), label="H2 rev cumulative [$]")
        else:
            # plot peaker energy
            months = range(len(blks_month))
            axs[2].step(months, storage_cumulated_mwh, label="Storage as peaker cumulative")
            axs[2].step(months, peaker_cumulated_mwh, label="Total peaker cumulative")
            axs2 = axs[2].twinx()
            axs2.step(months, avg_revenue_per_mwh, label="Avg Monthly Revenue [$/MWh]", color='k')
            axs2.legend(loc='center right')

        axs[2].legend()
        axs[2].grid(visible=True, which='major', color='k', linestyle='--', alpha=0.2)
        axs[2].minorticks_on()
        axs[2].grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.2)
        fig.tight_layout()

    plt.show()

    df = pd.DataFrame(index=range(n_time_points))
    df['Total Wind Generation [MW]'] = np.array(wind_gen) * 1e-3
    df['Total Power Output [MW]'] = np.sum((wind_out, batt_out, h2_turbine_elec), axis=0) * 1e-3
    df['Wind Power Output [MW]'] = np.array(wind_out) * 1e-3
    df['Wind Power to Battery [MW]'] = np.array(batt_in) * 1e-3
    df['State of Charge [MWh]'] = np.array(batt_soc) * 1e-3
    df['Battery Power Output [MW]'] = np.array(batt_out) * 1e-3
    df['Wind Power to PEM [MW]'] = np.array(wind_to_pem) * 1e-3
    df['PEM H2 Output [kg]'] = np.array(h2_prod)
    df['Tank H2 Input [kg]'] = np.array(h2_tank_in)
    df['H2 Sales [kg]'] = np.array(h2_sales)
    df['Turbine H2 Input [kg]"'] = np.array(h2_turbine_in)
    df['Tank Holdup [kg]'] = np.array(h2_tank_holdup)
    df['Turbine Power Output [MW]'] = np.array(h2_turbine_elec) * 1e-3
    if input_params['opt_mode'] == 'surrogate':
        df["Peaker Power [MW]"] = peaker_power
        ts_per_month = 8784 // 12
        df['Peaker Energy Price [$/MWh]'] = np.tile(avg_revenue_per_mwh, ts_per_month)
        df['Peaker CF Sold [1]'] = np.tile(cf_sold_month, ts_per_month)
        df['Peaker Revenue [$]'] = np.tile(revenue, ts_per_month)
        df['Peaker Delivered Power Cumulative [MWh]'] = np.tile(peaker_cumulated_mwh, ts_per_month)
        df['Peaker Dispatch CF Cumulative [1]'] = np.tile(cf_cumulative_month, ts_per_month)

    return design_res, df


if __name__ == "__main__":
    re_h2_parameters["pem_cap_cost"] *= 0.1
    re_h2_parameters["h2_price_per_kg"] = 0
    re_h2_parameters["turbine_cap_cost"] *= 0.1
    # re_h2_parameters["batt_cap_cost_kw"] *= 0.1
    # re_h2_parameters["batt_cap_cost_kwh"] *= 0.1
    re_h2_parameters["batt_mw"] = 0
    re_h2_parameters["batt_mwh"] = 0
    re_h2_parameters["turb_mw"] = 0
    re_h2_parameters['opt_mode'] = "surrogate"
    re_h2_parameters["tank_size"] = re_h2_parameters['turb_mw'] * 1e3 / re_h2_parameters['turb_conv']
    des_res, df_res = wind_battery_hydrogen_optimize(n_time_points=int(8784/2), input_params=re_h2_parameters, verbose=False, plot=False)
    df_res.to_parquet(re_h2_dir / "design_results.parquet")
    print(des_res)