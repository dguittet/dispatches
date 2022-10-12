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
from idaes.apps.grid_integration.model_data import ThermalGeneratorModelData
from idaes.apps.grid_integration import Tracker
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
    wind_cfs, wind_resource, loads_mw, wind_loads_mw = get_gen_outputs_from_rtsgmlc(wind_gen, gas_gen, reserves, shortfall, start_date)

    params["wind_mw"] = wind_gen_pmax
    params['batt_mwh'] = params['batt_mw'] * 4
    params["wind_resource"] = wind_resource
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
    model_data = ThermalGeneratorModelData(**generator_params)
    mp_model = MultiPeriodWindBatteryHydrogen(
        model_data, wind_cfs, params
    )
    return mp_model


def test_populate_model(mp_model):
    model = pyo.ConcreteModel()
    model.fs = pyo.Block()
    mp_model.populate_model(model.fs, 24)

    assert pyo.value(model.fs.P_T[0]) == 777.8
    assert pyo.value(model.fs.tot_cost[0]) == pytest.approx(7439.80, rel=1e-2)
    assert pyo.value(model.fs.wind_waste[0]) == 0


def test_update_model(mp_model):
    model = pyo.ConcreteModel()
    model.fs = pyo.Block()
    mp_model.populate_model(model.fs, 24)

    realized_soc = list(range(0, 24))
    realized_ep = [i / 2 for i in realized_soc]
    realized_holdup = list(range(0, 24))
    realized_throughput = [0 for i in realized_soc]

    mp_model.update_model(model.fs, realized_soc, realized_ep, realized_holdup, realized_throughput)

    active_blks = model.fs.windBatteryHydrogen.get_active_process_blocks()

    assert pyo.value(active_blks[0].fs.windpower.capacity_factor[0]) == pytest.approx(0.116, rel=1e-2)


def test_record_results(mp_model):
    model = pyo.ConcreteModel()
    model.fs = pyo.Block()
    mp_model.populate_model(model.fs, 24)
    mp_model.record_results(model.fs)

    assert len(mp_model.result_list)


def test_tracking(mp_model):
    solver = pyo.SolverFactory("cbc")
    tracker_object = Tracker(
        tracking_model_object=mp_model,
        tracking_horizon=24,
        n_tracking_hour=1,
        solver=solver,
    )

    dispatch = mp_model._design_params['load'][0:24]
    tracker_object.track_market_dispatch(dispatch, '2020-06-01', 0)

    assert tracker_object.result_list[0]['Power Underdelivered [MW]'].sum() == 0
    assert tracker_object.result_list[0]['Power Overdelivered [MW]'].sum() == 0
    for o, d in zip(tracker_object.result_list[0]['Power Output [MW]'].values, dispatch):
        assert o == pytest.approx(d, rel=1e-3)
