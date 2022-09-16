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
import pytest
import copy
import pyomo.environ as pyo
from dispatches.case_studies.renewables_h2_case.wind_battery_hydrogen_double_loop import MultiPeriodWindBatteryHydrogen
from dispatches.case_studies.renewables_h2_case.re_h2_parameters import re_h2_parameters, get_gen_outputs_from_rtsgmlc

@pytest.fixture
def mp_model():
    params = copy.copy(re_h2_parameters)
    wind_gen = "317_WIND"
    wind_gen_pmax = 799.1
    gas_gen = "317_CT"
    reserves = 10
    shortfall = 10000
    start_date = '2020-06-01 00:00:00'
    wind_capacity_factors, loads_mw = get_gen_outputs_from_rtsgmlc(wind_gen, gas_gen, reserves, shortfall, start_date)

    params["wind_mw"] = wind_gen_pmax
    params['batt_mwh'] = params['batt_mw'] * 4
    params["wind_resource"] = wind_capacity_factors
    params["load"] = loads_mw
    params["shortfall_price"] = shortfall

    hybrid_pmax = wind_gen_pmax + params['batt_mw'] + params['turb_mw']

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

    mp_model = MultiPeriodWindBatteryHydrogen(
        generator_params, wind_capacity_factors, wind_gen_pmax, params['batt_mw'], params['batt_mwh'],
        params['pem_mw'], params['tank_size'], params['turb_mw'], params['turb_conv']
    )
    return mp_model

def test_populate_model():
    params = copy.copy(re_h2_parameters)
    wind_gen = "317_WIND"
    wind_gen_pmax = 799.1
    gas_gen = "317_CT"
    reserves = 10
    shortfall = 10000
    start_date = '2020-06-01 00:00:00'
    wind_capacity_factors, loads_mw = get_gen_outputs_from_rtsgmlc(wind_gen, gas_gen, reserves, shortfall, start_date)

    params["wind_mw"] = wind_gen_pmax
    params['batt_mwh'] = params['batt_mw'] * 4
    params["wind_resource"] = wind_capacity_factors
    params["load"] = loads_mw
    params["shortfall_price"] = shortfall

    hybrid_pmax = wind_gen_pmax + params['batt_mw'] + params['turb_mw']

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

    mp_model = MultiPeriodWindBatteryHydrogen(generator_params, wind_capacity_factors, params)

    model = pyo.ConcreteModel()
    model.fs = pyo.Block()
    mp_model.populate_model(model.fs, 24)

test_populate_model()