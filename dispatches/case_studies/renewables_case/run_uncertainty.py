import numpy as np
import multiprocessing as mp
from pathlib import Path
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import numpy as np

from dispatches.case_studies.renewables_h2_case.wind_battery_hydrogen_flowsheet import wind_battery_hydrogen_optimize
from dispatches.case_studies.renewables_case.RE_flowsheet import default_input_params


file_dir = Path(__file__).parent / "Uncertainty_results_7550"
if not file_dir.exists():
    os.mkdir(file_dir)

price_cap = 7550
h2_price = 3
batt_mw = 357.1
batt_mwh = 375.89
pem_mw = 190.1
tank_tonh2 = 0
turb_mw = 0

syn_prices = []
xls = pd.ExcelFile(Path(__file__).parent / "ARMA_Model" / "synthetic_lmps_7550.xlsx")
for name in xls.sheet_names:
    syn_prices.append(np.clip(pd.read_excel(xls, name)[2018].values, 0, price_cap))

def run_design(n, prices):
    input_params = default_input_params.copy()
    input_params['design_opt'] = False
    input_params['build_add_wind'] = 0
    input_params['opt_mode'] = "pricetaker"
    input_params['h2_price_per_kg'] = h2_price
    input_params["batt_mw"] = batt_mw
    input_params["batt_mwh"] = batt_mwh
    input_params["turb_conv"] = 20
    input_params["tank_size"] = tank_tonh2
    input_params["turb_mw"] = turb_mw
    input_params['LMPs'] = prices[24:]      # start on day 2 like original conceptual design runs
    des_res = wind_battery_hydrogen_optimize(
        # n_time_points=24 * 7, 
        n_time_points=len(input_params['LMPs']), 
        input_params=input_params, verbose=False, plot=False)
    res = {**input_params, **des_res[0]}
    res.pop("LMPs")
    res.pop("design_opt")
    res.pop("extant_wind")
    res.pop("wind_resource")
    res.pop("pyo_model")

    with open(file_dir / f"result_{n}.json", 'w') as f:
        json.dump(res, f)
    print(f"Finished: {n}")
    des_res[1].to_parquet(file_dir / f"timeseries_results_{n}.parquet")
    return res, des_res[1]

inputs = enumerate(syn_prices)

with mp.Pool(processes=10) as p:
    res = p.starmap(run_design, inputs)