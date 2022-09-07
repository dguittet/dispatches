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
import platform
from idaes.core.util.model_statistics import degrees_of_freedom

from dispatches.case_studies.renewables_case.RE_flowsheet import *
from dispatches.case_studies.renewables_case.wind_battery_LMP import wind_battery_optimize, record_results, plot_results
from dispatches.case_studies.renewables_h2_case.wind_battery_hydrogen_flowsheet import wind_battery_hydrogen_optimize
from dispatches.case_studies.renewables_h2_case.re_h2_parameters import re_h2_parameters, get_gen_outputs_from_rtsgmlc

@pytest.fixture
def input_params():
    params = copy.copy(re_h2_parameters)
    wind_gen = "317_WIND"
    wind_gen_pmax = 799.1
    gas_gen = "317_CT"
    reserves = 10
    shortfall = 10000
    start_date = '2020-06-01 00:00:00'
    wind_capacity_factors, loads_mw = get_gen_outputs_from_rtsgmlc(wind_gen, gas_gen, reserves, shortfall, start_date)

    params["wind_mw"] = wind_gen_pmax
    params["h2_price_per_kg"] = 0
    params["extant_wind"] = False
    params["wind_resource"] = wind_capacity_factors
    params["load"] = loads_mw
    params["shortfall_price"] = shortfall
    return params

def test_wind_battery_optimize(input_params):
    mp = wind_battery_optimize(n_time_points=7 * 24, input_params=input_params, verbose=True)
    blks = mp.get_active_process_blocks()
    assert sum(value(blk.under_power) for blk in blks) == 0
    assert value(mp.pyomo_model.NPV) == pytest.approx(-1691292641, rel=1e-3)
    assert value(mp.pyomo_model.annual_revenue) == pytest.approx(-24628238, rel=1e-3)
    assert value(mp.pyomo_model.battery_system_capacity) == pytest.approx(40254, rel=1e-3)
    assert value(mp.pyomo_model.wind_system_capacity) == pytest.approx(815735, rel=1e-3)
    plot_results(*record_results(mp), input_params['opt_mode'])


def test_wind_battery_hydrogen_optimize(input_params):
    design_res = wind_battery_hydrogen_optimize(7 * 24, input_params, verbose=False, plot=False)
    assert design_res['batt_mw'] == pytest.approx(32.99, rel=1e-2)
    assert design_res['pem_mw'] == pytest.approx(0, abs=3)
    assert design_res['tank_tonH2'] == pytest.approx(0, abs=3)
    assert design_res['turb_mw'] == pytest.approx(0, abs=3)
    assert design_res['annual_under_power'] == pytest.approx(0, abs=1)
    assert design_res['annual_rev_h2'] == pytest.approx(0, abs=1)
    assert design_res['annual_rev_E'] == pytest.approx(-34339507, rel=1e-2)
    assert design_res['NPV'] == pytest.approx(-1807539055, rel=1e-2)


def test_wind_battery_hydrogen_optimize_cheap_hydrogen(input_params):
    input_params["pem_cap_cost"] *= 0.1
    input_params["tank_cap_cost_per_kg"] *= 0.1
    input_params["turbine_cap_cost"] *= 0.1
    design_res = wind_battery_hydrogen_optimize(7 * 24, input_params, verbose=False, plot=False)
    assert design_res['batt_mw'] == pytest.approx(20, rel=1e-2)
    assert design_res['pem_mw'] == pytest.approx(2, abs=3)
    assert design_res['tank_tonH2'] == pytest.approx(1.54, abs=3)
    assert design_res['turb_mw'] == pytest.approx(12.99, abs=3)
    assert design_res['annual_rev_h2'] == pytest.approx(0, abs=1)
    assert design_res['annual_rev_E'] == pytest.approx(-34842617, rel=1e-2)
    assert design_res['NPV'] == pytest.approx(-1801460534, rel=1e-2)
