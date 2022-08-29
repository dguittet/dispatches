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
import numpy as np
import copy
from pathlib import Path
import pandas as pd
from pyomo.common.fileutils import this_file_dir

from dispatches_sample_data import rts_gmlc
from dispatches.case_studies.renewables_case.double_loop_utils import read_rts_gmlc_wind_inputs_with_fix

re_h2_dir = Path(this_file_dir())

timestep_hrs = 1                            # timestep [hr]
# constants
h2_mols_per_kg = 500
H2_mass = 2.016 / 1000

# costs in per kW unless specified otherwise
wind_cap_cost = 1550
wind_op_cost = 43
batt_cap_cost = 300 * 4                     # per kW for 4 hour battery
batt_rep_cost_kwh = batt_cap_cost * 0.5 / 4 # assume 50% price w/ discounting and 4 hour battery
pem_cap_cost = 1630
pem_op_cost = 47.9
pem_var_cost = 1.3/1000                     # per kWh
tank_cap_cost_per_m3 = 29 * 0.8 * 1000      # per m^3
tank_cap_cost_per_kg = 29 * 33.5            # per kg
tank_op_cost = .17 * tank_cap_cost_per_kg   # per kg
turbine_cap_cost = 1000
turbine_op_cost = 11.65
turbine_var_cost = 4.27/1000                # per kWh

# prices
h2_price_per_kg = 2

# sizes
fixed_wind_mw = 847
wind_mw_ub = 10000
fixed_batt_mw = 4874
fixed_pem_mw = 643
turb_p_mw = 1
valve_cv = 0.00001
fixed_tank_size = 0.5

# operation parameters
pem_bar = 1.01325
pem_temp = 300                  # [K]
# battery_ramp_rate = 25 * 1e3              # kwh/hr
battery_ramp_rate = 1e8
h2_turb_min_flow = 1e-3
air_h2_ratio = 10.76
compressor_dp = 24.01
max_pressure_bar = 700

# simple financial assumptions
i = 0.05                                    # discount rate
N = 30                                      # years
PA = ((1+i)**N - 1)/(i*(1+i)**N)            # present value / annuity = 1 / CRF

wind_gen = "317_WIND"
wind_gen_pmax = 799.1
gas_gen = "317_CT"
market = "RT"
reserves = 10
shortfall = 10000

# load pre-compiled RTS-GMLC output data
df = pd.read_csv(re_h2_dir / "data" / "Wind_Thermal_Gen.csv", index_col="Datetime", parse_dates=True)
df = df.query(f"Reserves == {reserves} & Shortfall == {shortfall}")
df = df[df['Mod Gen'] == gas_gen]

if not len(df):
    raise ValueError

wind_cfs = df['317_WIND_1-RTCF'].values
wind_capacity_factors = {t:
                            {'wind_resource_config': {
                                'capacity_factor': 
                                    [wind_cfs[t]]}} for t in range(len(wind_cfs))}

loads_mw = (df[f"{wind_gen} Output"] + df[f"Gas Output"]).values

default_input_params = {
    "wind_mw": fixed_wind_mw,
    "wind_mw_ub": wind_mw_ub,
    "batt_mw": fixed_batt_mw,
    "pem_mw": fixed_pem_mw,
    "pem_bar": pem_bar,
    "pem_temp": pem_temp,
    "tank_size": fixed_tank_size,
    "tank_type": "simple",
    "turb_mw": turb_p_mw,

    "wind_resource": wind_capacity_factors,
    "h2_price_per_kg": h2_price_per_kg,

    "design_opt": True,
    "extant_wind": False,

    "opt_mode": "meet_load",
    "load": loads_mw
}
