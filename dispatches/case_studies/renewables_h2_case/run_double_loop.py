import copy
import pyomo.environ as pyo
from idaes.apps.grid_integration.model_data import ThermalGeneratorModelData
from idaes.apps.grid_integration import Tracker
from dispatches.case_studies.renewables_h2_case.wind_battery_hydrogen_double_loop import MultiPeriodWindBatteryHydrogen
from dispatches.case_studies.renewables_h2_case.re_h2_parameters import re_h2_parameters, get_gen_outputs_from_rtsgmlc

params = copy.copy(re_h2_parameters)
wind_gen = "317_WIND"
wind_gen_pmax = 799.1
gas_gen = "317_CT"
reserves = 15
shortfall = 500
start_date = '2020-01-01 00:00:00'
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
params["h2_price_per_kg"] = h2_price_per_kg

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