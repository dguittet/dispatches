from prescient.simulator import Prescient
from types import ModuleType
from optparse import OptionParser
from wind_battery_double_loop import MultiPeriodWindBattery
from idaes.apps.grid_integration import (
    Tracker,
    DoubleLoopCoordinator,
    Bidder,
    SelfScheduler,
)
from idaes.apps.grid_integration.forecaster import Backcaster
from idaes.apps.grid_integration.model_data import (
    RenewableGeneratorModelData,
    ThermalGeneratorModelData,
)
import pyomo.environ as pyo
import pandas as pd
import os

this_file_path = os.path.dirname(os.path.realpath(__file__))

usage = "Run double loop simulation with RE model."
parser = OptionParser(usage)

parser.add_option(
    "--sim_id",
    dest="sim_id",
    help="Indicate the simulation ID.",
    action="store",
    type="int",
    default=0,
)

parser.add_option(
    "--wind_pmax",
    dest="wind_pmax",
    help="Set wind capacity in MW.",
    action="store",
    type="float",
    default=200.0,
)

parser.add_option(
    "--battery_energy_capacity",
    dest="battery_energy_capacity",
    help="Set the battery energy capacity in MWh.",
    action="store",
    type="float",
    default=100.0,
)

parser.add_option(
    "--battery_pmax",
    dest="battery_pmax",
    help="Set the battery power capacity in MW.",
    action="store",
    type="float",
    default=25.0,
)

parser.add_option(
    "--n_scenario",
    dest="n_scenario",
    help="Set the number of price scenarios.",
    action="store",
    type="int",
    default=3,
)

parser.add_option(
    "--reserve_factor",
    dest="reserve_factor",
    help="Set the reserve factor.",
    action="store",
    type="float",
    default=0.0,
)

parser.add_option(
    "--participation_mode",
    dest="participation_mode",
    help="Indicate the market participation mode.",
    action="store",
    type="str",
    default="Bid",
)

(options, args) = parser.parse_args()

sim_id = options.sim_id
wind_pmax = options.wind_pmax
battery_energy_capacity = options.battery_energy_capacity
battery_pmax = options.battery_pmax
n_scenario = options.n_scenario
participation_mode = options.participation_mode
reserve_factor = options.reserve_factor

allowed_participation_modes = {"Bid", "SelfSchedule"}
if participation_mode not in allowed_participation_modes:
    raise ValueError(
        f"The provided participation mode {participation_mode} is not supported."
    )

p_min = 0
default_wind_bus = 309
bus_name = "Carter"
wind_generator = "309_WIND_1"
capacity_factor_df = pd.read_csv(os.path.join(this_file_path, "capacity_factors.csv"))
gen_capacity_factor = list(capacity_factor_df[wind_generator])[24:]

# NOTE: `rts_gmlc_data_dir` should point to a directory containing RTS-GMLC scenarios
rts_gmlc_data_dir = "/afs/crc.nd.edu/user/x/xgao1/DowlingLab/RTS-GMLC/RTS_Data/SourceData"
output_dir = f"sim_{sim_id}_results"

solver = pyo.SolverFactory("gurobi")

if participation_mode == "Bid":
    thermal_generator_params = {
        "gen_name": wind_generator,
        "bus": bus_name,
        "p_min": p_min,
        "p_max": wind_pmax,
        "min_down_time": 0,
        "min_up_time": 0,
        "ramp_up_60min": wind_pmax + battery_pmax,
        "ramp_down_60min": wind_pmax + battery_pmax,
        "shutdown_capacity": wind_pmax + battery_pmax,
        "startup_capacity": 0,
        "initial_status": 1,
        "initial_p_output": 0,
        "production_cost_bid_pairs": [(p_min, 0), (wind_pmax, 0)],
        "startup_cost_pairs": [(0, 0)],
        "fixed_commitment": None,
    }
    model_data = ThermalGeneratorModelData(**thermal_generator_params)
elif participation_mode == "SelfSchedule":
    generator_params = {
        "gen_name": wind_generator,
        "bus": bus_name,
        "p_min": p_min,
        "p_max": wind_pmax,
        "p_cost": 0,
        "fixed_commitment": None,
    }
    model_data = RenewableGeneratorModelData(**generator_params)

historical_da_prices = {
    bus_name: [
        19.983547,
        0.0,
        0.0,
        19.983547,
        21.647258,
        21.647258,
        33.946708,
        21.647258,
        0.0,
        0.0,
        19.983547,
        20.846138,
        20.419098,
        21.116411,
        21.116411,
        21.843654,
        33.752662,
        27.274616,
        27.274616,
        26.324557,
        23.128644,
        21.288154,
        21.116714,
        21.116714,
    ]
}
historical_rt_prices = {
    bus_name: [
        30.729141,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        32.451804,
        34.888412,
        0.0,
        0.0,
        0.0,
        0.0,
        19.983547,
        21.116411,
        19.034775,
        16.970947,
        20.419098,
        26.657418,
        25.9087,
        24.617414,
        24.617414,
        22.492854,
        10000.0,
        23.437807,
    ]
}

################################################################################
################################# bidder #######################################
################################################################################
day_ahead_horizon = 48
real_time_horizon = 4

mp_wind_battery_bid = MultiPeriodWindBattery(
    model_data=model_data,
    wind_capacity_factors=gen_capacity_factor,
    wind_pmax_mw=wind_pmax,
    battery_pmax_mw=battery_pmax,
    battery_energy_capacity_mwh=battery_energy_capacity,
)

backcaster = Backcaster(historical_da_prices, historical_rt_prices)

if participation_mode == "Bid":
    bidder_object = Bidder(
        bidding_model_object=mp_wind_battery_bid,
        day_ahead_horizon=day_ahead_horizon,
        real_time_horizon=real_time_horizon,
        n_scenario=n_scenario,
        solver=solver,
        forecaster=backcaster,
    )
elif participation_mode == "SelfSchedule":
    bidder_object = SelfScheduler(
        bidding_model_object=mp_wind_battery_bid,
        day_ahead_horizon=day_ahead_horizon,
        real_time_horizon=real_time_horizon,
        n_scenario=n_scenario,
        solver=solver,
        forecaster=backcaster,
    )

################################################################################
################################# Tracker ######################################
################################################################################

tracking_horizon = 4
n_tracking_hour = 1

mp_wind_battery_track = MultiPeriodWindBattery(
    model_data=model_data,
    wind_capacity_factors=gen_capacity_factor,
    wind_pmax_mw=wind_pmax,
    battery_pmax_mw=battery_pmax,
    battery_energy_capacity_mwh=battery_energy_capacity,
)

# create a `Tracker` using`mp_wind_battery`
tracker_object = Tracker(
    tracking_model_object=mp_wind_battery_track,
    tracking_horizon=tracking_horizon,
    n_tracking_hour=n_tracking_hour,
    solver=solver,
)

mp_wind_battery_track_project = MultiPeriodWindBattery(
    model_data=model_data,
    wind_capacity_factors=gen_capacity_factor,
    wind_pmax_mw=wind_pmax,
    battery_pmax_mw=battery_pmax,
    battery_energy_capacity_mwh=battery_energy_capacity,
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
    "start_date": "01-02-2020",
    "num_days": 364,
    "sced_horizon": 4,
    "ruc_horizon": 48,
    "compute_market_settlements": True,
    "day_ahead_pricing": "LMP",
    "ruc_mipgap": 0.05,
    "symbolic_solver_labels": True,
    "reserve_factor": reserve_factor,
    "deterministic_ruc_solver": "gurobi",
    "sced_solver": "gurobi",
    "plugin": {
        "doubleloop": {
            "module": plugin_module,
            "bidding_generator": "309_WIND_1",
        }
    },
}

Prescient().simulate(**prescient_options)

# write options into the result folder
with open(os.path.join(output_dir, "sim_options.txt"), "w") as f:
    f.write(str(options))
