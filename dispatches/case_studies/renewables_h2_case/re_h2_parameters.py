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
from dispatches.case_studies.renewables_case.load_parameters import *

re_h2_dir = Path(this_file_dir())

turb_conv_rate = 20                         # kwh/kgH2 for h2 turbine

def get_gen_outputs_from_rtsgmlc(wind_gen, gas_gen, reserves, shortfall, start_date=None):
    """
    Load pre-compiled RTS-GMLC output data
    All real-time market data
    """
    df = pd.read_csv(re_h2_dir / "data" / "Wind_Thermal_Gen.csv", index_col="Datetime", parse_dates=True)
    df = df.query(f"Reserves == {reserves} & Shortfall == {shortfall}")
    df = df[df['Mod Gen'] == gas_gen]
    if start_date is not None:
        df = df[df.index >= start_date]

    if not len(df):
        raise ValueError

    wind_cfs = df['317_WIND_1-RTCF'].values
    wind_resource = {t:
                                {'wind_resource_config': {
                                    'capacity_factor': 
                                        [wind_cfs[t]]}} for t in range(len(wind_cfs))}

    loads_mw = (df[f"{wind_gen} Output"] + df[f"Gas Output"]).values
    return wind_cfs, wind_resource, loads_mw

wind_gen = "317_WIND"
wind_gen_pmax = 799.1
gas_gen = "317_CT"
reserves = 15
shortfall = 500
start_date = '2020-01-01 00:00:00'
wind_cfs, wind_resource, loads_mw = get_gen_outputs_from_rtsgmlc(wind_gen, gas_gen, reserves, shortfall, start_date)


re_h2_parameters = {
    "wind_mw": wind_gen_pmax,
    "wind_mw_ub": wind_mw_ub,
    "batt_mw": fixed_batt_mw,
    "pem_mw": fixed_pem_mw,
    "pem_bar": pem_bar,
    "pem_temp": pem_temp,
    "tank_size": fixed_tank_size,
    "turb_mw": turb_p_mw,
    "turb_conv": turb_conv_rate,

    "wind_resource": wind_resource,
    "h2_price_per_kg": h2_price_per_kg,

    "build_add_wind": True,

    "opt_mode": "meet_load",
    "load": loads_mw,
    'shortfall_price': shortfall,

    "wind_cap_cost": wind_cap_cost,
    "wind_op_cost": wind_op_cost,
    "batt_cap_cost_kw": batt_cap_cost_kw,
    "batt_cap_cost_kwh": batt_cap_cost_kwh,
    "batt_rep_cost_kwh": batt_rep_cost_kwh,
    "pem_cap_cost": pem_cap_cost,
    "pem_op_cost": pem_op_cost,
    "pem_var_cost": pem_var_cost,
    "tank_cap_cost_per_kg": tank_cap_cost_per_kg,
    "tank_op_cost":tank_op_cost,
    "turbine_cap_cost": turbine_cap_cost,
    "turbine_op_cost": turbine_op_cost,
    "turbine_var_cost": turbine_var_cost
}
