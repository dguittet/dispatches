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
from re import A
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
    _, wind_capacity_factors, loads_mw, wind_loads_mw = get_gen_outputs_from_rtsgmlc(wind_gen, gas_gen, reserves, shortfall, start_date)

    params["wind_mw"] = wind_gen_pmax
    params["wind_resource"] = wind_capacity_factors
    params["load"] = loads_mw
    params["wind_load"] = wind_loads_mw
    params["shortfall_price"] = shortfall
    return params

def test_wind_battery_optimize(input_params):
    input_params['design_opt'] = True
    input_params['extant_wind'] = True
    mp = wind_battery_optimize(n_time_points=7 * 24, input_params=input_params, verbose=True)
    blks = mp.get_active_process_blocks()
    assert sum(value(blk.under_power) for blk in blks) == 0
    assert value(mp.pyomo_model.NPV) == pytest.approx(-571293680, rel=1e-3)
    assert value(mp.pyomo_model.annual_revenue) == pytest.approx(-34326140, rel=1e-3)
    assert value(mp.pyomo_model.battery_system_capacity) == pytest.approx(33000, rel=1e-3)
    assert value(mp.pyomo_model.battery_system_energy) == pytest.approx(132000, rel=1e-3)
    assert value(mp.pyomo_model.wind_system_capacity) == pytest.approx(799100, rel=1e-3)
    plot_results(*record_results(mp), input_params['opt_mode'])


def test_wind_battery_hydrogen_optimize(input_params):
    design_res, _ = wind_battery_hydrogen_optimize(int(8760/12), input_params, verbose=False, plot=True)
    assert design_res['wind_mw'] == pytest.approx(799.1, rel=1e-2)
    assert design_res['batt_mw'] == pytest.approx(85.44, rel=1e-2)
    assert design_res['batt_mwh'] == pytest.approx(259.284, rel=1e-2)
    assert design_res['pem_mw'] == pytest.approx(2, abs=1)
    assert design_res['tank_tonH2'] == pytest.approx(16.7, abs=2)
    assert design_res['turb_mw'] == pytest.approx(2.56, abs=1)
    assert design_res['annual_under_power'] == pytest.approx(0, abs=1)
    assert design_res['annual_rev_h2'] == pytest.approx(0, abs=1)
    assert design_res['annual_costs_E'] == pytest.approx(28483, rel=1e-2)
    assert design_res['NPV'] == pytest.approx(-662713110, rel=1e-2)


def test_wind_battery_hydrogen_optimize_cheap_hydrogen(input_params):
    input_params["turbine_cap_cost"] *= 0.1
    design_res, _ = wind_battery_hydrogen_optimize(int(8760/12), input_params, verbose=False, plot=False)
    assert design_res['wind_mw'] == pytest.approx(808.7, rel=1e-2)
    assert design_res['batt_mw'] == pytest.approx(2.30, rel=5e-2)
    assert design_res['batt_mwh'] == pytest.approx(13, abs=1)
    assert design_res['pem_mw'] == pytest.approx(6.7, abs=1)
    assert design_res['tank_tonH2'] == pytest.approx(27.4, abs=3)
    assert design_res['turb_mw'] == pytest.approx(85.63, abs=3)
    assert design_res['annual_rev_h2'] == pytest.approx(0, abs=1)
    assert design_res['annual_costs_E'] == pytest.approx(54212, rel=1e-2)
    assert design_res['NPV'] == pytest.approx(-632225328, rel=1e-2)
    assert design_res["capital_cost"] == pytest.approx(50405415, rel=1e-2)
