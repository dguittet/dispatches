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
from prescient.simulator import Prescient
from types import ModuleType
from argparse import ArgumentParser
from wind_battery_hydrogen_double_loop import MultiPeriodWindBatteryHydrogen
from re_h2_parameters import *
from parametrized_bidder import FixedParametrizedBidder, PerfectForecaster
from idaes.apps.grid_integration import (
    Tracker,
    DoubleLoopCoordinator
)
from idaes.apps.grid_integration.forecaster import Backcaster
from idaes.apps.grid_integration.model_data import ThermalGeneratorModelData
import pyomo.environ as pyo
from pyomo.common.fileutils import this_file_dir
import pandas as pd
from pathlib import Path
from dispatches_sample_data import rts_gmlc


this_file_path = Path(this_file_dir())

usage = "Run double loop simulation with RE model."
parser = ArgumentParser(usage)

parser.add_argument(
    "--sim_id",
    dest="sim_id",
    help="Indicate the simulation ID.",
    action="store",
    type=int,
    default=0,
)

parser.add_argument(
    "--wind_pmax",
    dest="wind_pmax",
    help="Set wind capacity in MW.",
    action="store",
    type=float,
    default=wind_gen_pmax,
)

parser.add_argument(
    "--battery_energy_capacity",
    dest="battery_energy_capacity",
    help="Set the battery energy capacity in MWh.",
    action="store",
    type=float,
    default=100.0,
)

parser.add_argument(
    "--battery_pmax",
    dest="battery_pmax",
    help="Set the battery power capacity in MW.",
    action="store",
    type=float,
    default=25.0,
)

parser.add_argument(
    "--storage_bid",
    dest="storage_bid",
    help="Set the storage bid price in $/MW.",
    action="store",
    type=float,
    default=50.0,
)

parser.add_argument(
    "--reserve_factor",
    dest="reserve_factor",
    help="Set the reserve factor.",
    action="store",
    type=float,
    default=reserves * 1e-2,
)

options = parser.parse_args()

sim_id = options.sim_id
wind_pmax = options.wind_pmax
battery_energy_capacity = options.battery_energy_capacity
battery_pmax = options.battery_pmax
storage_bid = options.storage_bid
reserve_factor = options.reserve_factor

battery_pmax = 6
battery_energy_capacity = 106
turb_p_mw = 0.14
wind_pmax = 800.6
storage_bid = 96.25

hybrid_pmax = wind_pmax + battery_pmax + turb_p_mw
p_min = 0
default_wind_bus = 317
bus_name = "Chuhsi"
wind_generator = "317_WIND_1"
start_date = "01-01-2020"
input_params = re_h2_parameters.copy()
input_params['batt_mw'] = battery_pmax
input_params['batt_mwh'] = battery_energy_capacity
input_params['batt_hr'] = battery_energy_capacity / battery_pmax
input_params['wind_mw'] = wind_pmax
input_params['tank_size'] = 0.4044 / kg_to_tons
input_params['pem_mw'] = 0.26
input_params['turb_mw'] = turb_p_mw
input_params['turb_conv'] = 15
input_params['h2_price_per_kg'] = 1

wind_cfs, wind_resource, loads_mw, wind_loads_mw = get_gen_outputs_from_rtsgmlc(wind_gen, gas_gen, reserves, shortfall, start_date)

# NOTE: `rts_gmlc_data_dir` should point to a directory containing RTS-GMLC scenarios
rts_gmlc_data_dir = rts_gmlc.source_data_path
output_dir = Path(f"double_loop_parametrized_rdc_results")

solver = pyo.SolverFactory("xpress_direct")

thermal_generator_params = {
    "gen_name": wind_generator,
    "bus": bus_name,
    "p_min": p_min,
    "p_max": hybrid_pmax,
    "min_down_time": 0,
    "min_up_time": 0,
    "ramp_up_60min": hybrid_pmax,
    "ramp_down_60min": hybrid_pmax,
    "shutdown_capacity": hybrid_pmax,
    "startup_capacity": hybrid_pmax,
    "initial_status": 1,
    "initial_p_output": 0,
    "production_cost_bid_pairs": [(p_min, 0), (hybrid_pmax, 0)],
    "startup_cost_pairs": [(0, 0)],
    "fixed_commitment": None,
}
model_data = ThermalGeneratorModelData(**thermal_generator_params)


################################################################################
################################# bidder #######################################
################################################################################
day_ahead_horizon = 48
real_time_horizon = 4

forecaster = PerfectForecaster(re_h2_dir / "data" / "Wind_Thermal_Gen.csv")

mp_wind_battery_bid = MultiPeriodWindBatteryHydrogen(
    model_data=model_data,
    wind_capacity_factors=wind_cfs,
    input_params=input_params
)

bidder_object = FixedParametrizedBidder(
    bidding_model_object=mp_wind_battery_bid,
    day_ahead_horizon=day_ahead_horizon,
    real_time_horizon=real_time_horizon,
    n_scenario=1,
    solver=solver,
    forecaster=forecaster,
    storage_marginal_cost=storage_bid,
    storage_mw=battery_pmax + turb_p_mw
)

################################################################################
################################# Tracker ######################################
################################################################################

tracking_horizon = 4
n_tracking_hour = 1

mp_wind_battery_track = MultiPeriodWindBatteryHydrogen(
    model_data=model_data,
    wind_capacity_factors=wind_cfs,
    input_params=input_params
)

# create a `Tracker` using`mp_wind_battery`
tracker_object = Tracker(
    tracking_model_object=mp_wind_battery_track,
    tracking_horizon=tracking_horizon,
    n_tracking_hour=n_tracking_hour,
    solver=solver,
)

mp_wind_battery_track_project = MultiPeriodWindBatteryHydrogen(
    model_data=model_data,
    wind_capacity_factors=wind_cfs,
    input_params=input_params
)

# create a `Tracker` using`mp_wind_battery`
project_tracker_object = Tracker(
    tracking_model_object=mp_wind_battery_track_project,
    tracking_horizon=tracking_horizon,
    n_tracking_hour=n_tracking_hour,
    solver=solver,
)

################################################################################
################################# Coordinator ##################################
################################################################################

coordinator = DoubleLoopCoordinator(
    bidder=bidder_object,
    tracker=tracker_object,
    projection_tracker=project_tracker_object,
)


class PrescientPluginModule(ModuleType):
    def __init__(self, get_configuration, register_plugins):
        self.get_configuration = get_configuration
        self.register_plugins = register_plugins


plugin_module = PrescientPluginModule(
    get_configuration=coordinator.get_configuration,
    register_plugins=coordinator.register_plugins,
)

prescient_options = {
    "data_path": rts_gmlc_data_dir,
    "input_format": "rts-gmlc",
    "simulate_out_of_sample": True,
    "run_sced_with_persistent_forecast_errors": True,
    "output_directory": output_dir,
    "monitor_all_contingencies":False,
    "start_date": start_date,
    "num_days": 364,
    "sced_horizon": 1,
    "ruc_horizon": 36,
    "compute_market_settlements": True,
    "day_ahead_pricing": "aCHP",
    "ruc_mipgap": 0.01,
    "symbolic_solver_labels": True,
    "reserve_factor": reserve_factor,
    "price_threshold": shortfall,
    "transmission_price_threshold": shortfall / 2,
    "reserve_price_threshold": shortfall / 10,
    # "deterministic_ruc_solver": "xpress_direct",
    "deterministic_ruc_solver": "xpress_persistent",
    "deterministic_ruc_solver_options" : {"threads":2, "heurstrategy":2, "cutstrategy":3, "symmetry":2, "maxnode":1000},
    "sced_solver": "xpress_persistent",
    "enforce_sced_shutdown_ramprate":False,
    "ruc_slack_type":"ref-bus-and-branches",
    "sced_slack_type":"ref-bus-and-branches",
    "disable_stackgraphs":True,
    "plugin": {
        "doubleloop": {
            "module": plugin_module,
            "bidding_generator": wind_generator,
        }
    },
}

Prescient().simulate(**prescient_options)

# write options into the result folder
with open(output_dir / "sim_options.json", "w") as f:
    f.write(str(input_params))
