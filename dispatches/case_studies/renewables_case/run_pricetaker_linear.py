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
import multiprocessing as mp
from itertools import product
import os
import copy
import json
from pyomo.common.tempfiles import TempfileManager
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dispatches.case_studies.renewables_h2_case.wind_battery_hydrogen_flowsheet import wind_battery_hydrogen_optimize
from dispatches.case_studies.renewables_case.RE_flowsheet import default_input_params, market, prices


# TempfileManager.tempdir = '/tmp/scratch'
file_dir = Path(__file__).parent / "results_PT_4"
if not file_dir.exists():
    os.mkdir(file_dir)

build_add_wind = 0 # if False, wind size is fixed. Either way, all wind capacity is part of capital cost

def run_design(h2_price, price_cap):
    input_params = default_input_params.copy()
    input_params['build_add_wind'] = build_add_wind
    input_params['opt_mode'] = "pricetaker"
    input_params['h2_price_per_kg'] = h2_price
    input_params["turb_conv"] = 20
    input_params["batt_hr"] = 4
    input_params["tank_size"] = input_params['turb_mw'] * 1e3 / input_params['turb_conv']
    input_params['LMPs'] = copy.copy(prices)
    input_params['LMPs'][input_params['LMPs'] > price_cap] = price_cap
    if (file_dir / f"result_{market}_{h2_price}_{price_cap}_{build_add_wind}.json").exists():
        with open(file_dir / f"result_{market}_{h2_price}_{price_cap}_{build_add_wind}.json", 'r') as f:
            res = json.load(f)
        res = {**input_params, **res}
        res.pop("LMPs")
        res.pop("design_opt")
        res.pop("extant_wind")
        res.pop("wind_resource")
        print(f"Already complete: {h2_price} {price_cap} {build_add_wind}")
        return res
    print(f"Running: {h2_price} {price_cap} {build_add_wind}")
    des_res = wind_battery_hydrogen_optimize(
        # n_time_points=24 * 7, 
        n_time_points=len(prices), 
        input_params=input_params, verbose=False, plot=False)
    res = {**input_params, **des_res[0]}
    res.pop("LMPs")
    res.pop("design_opt")
    res.pop("extant_wind")
    res.pop("wind_resource")
    res.pop("pyo_model")
    with open(file_dir / f"result_{market}_{h2_price}_{price_cap}_{build_add_wind}.json", 'w') as f:
        json.dump(res, f)
    print(f"Finished: {h2_price} {price_cap} {build_add_wind}")
    des_res[1].to_parquet(file_dir / f"timeseries_results_{h2_price}_{price_cap}_{build_add_wind}.parquet")
    return res

# run_design(2.2, 200.)
# exit()

print(f"Writing to 'design_{market}_{build_add_wind}_results.csv'")
h2_prices = np.linspace(2, 3, 5)
price_cap = np.linspace(200, 10000, 5)
# h2_prices = np.flip(h2_prices)
# price_cap = np.flip(price_cap)
inputs = product(h2_prices, price_cap)

with mp.Pool(processes=2) as p:
    res = p.starmap(run_design, inputs)

df = pd.DataFrame(res)
df.to_csv(file_dir / f"design_{market}_{build_add_wind}_results.csv")
