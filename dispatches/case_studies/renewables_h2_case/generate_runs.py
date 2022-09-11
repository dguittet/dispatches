import json
import os, sys
from pathlib import Path
import numpy as np
from itertools import product
from dispatches.case_studies.renewables_h2_case.re_h2_parameters import re_h2_parameters, get_gen_outputs_from_rtsgmlc


wind_gen = "317_WIND"
wind_gen_pmax = 799.1
gas_gen = "317_CT"
reserves = 15
shortfall = 500
start_date = '2020-01-01 00:00:00'

n_samples = 4
h2_prices = np.linspace(0, 3, n_samples)
batt_kw_costs = np.linspace(300, 900, n_samples)
batt_kwh_cost_ratios = np.linspace(.2, .4, n_samples)
pem_cap_costs = np.linspace(1200, 2000, n_samples)
tank_cap_costs = np.linspace(375, 625, n_samples)
turbine_cap_costs = np.linspace(750, 1250, n_samples)

turb_conv_rates = np.linspace(10, 25, n_samples)

all_runs = list(product(h2_prices, batt_kw_costs, batt_kwh_cost_ratios, pem_cap_costs, tank_cap_costs, turbine_cap_costs, turb_conv_rates))
print(len(all_runs))
params = re_h2_parameters.copy()
params.pop("load")
params.pop("wind_resource")
params['wind_gen'] = wind_gen
params['wind_gen_pmax'] = wind_gen_pmax
params['gas_gen'] = gas_gen
params['reserves'] = reserves
params['shortfall'] = shortfall
params['start_date'] = start_date


output_dir = Path(__file__).absolute().parent / f"{len(all_runs)}_results_{317}_{gas_gen[-2:]}_{reserves}_{shortfall}"
if not output_dir.exists():
    os.mkdir(output_dir)

shell_cmds = f"""#!/bin/bash
{sys.executable} {output_dir / "runner.py"} """

jade_file = {
  "configuration_class": "GenericCommandConfiguration",
  "configuration_module": "jade.extensions.generic_command.generic_command_configuration",
  "format_version": "v0.2.0",
  "jobs": []}

for n, (h2_price, batt_kw_cost, batt_kwh_cost_ratio, pem_cap_cost, tank_cap_cost, turbine_cap_cost, turb_conv_rate) in enumerate(all_runs):
    params["h2_price_per_kg"] = h2_price
    params["batt_cap_cost_kw"] = batt_kw_cost
    params["batt_cap_cost_kwh"] = batt_kw_cost * batt_kwh_cost_ratio
    params["pem_cap_cost"] = pem_cap_cost
    params["tank_cap_cost_per_kg"] = tank_cap_cost
    params["turbine_cap_cost"] = turbine_cap_cost
    params["turb_conv"] = turb_conv_rate
    file_name = output_dir / f"params_{n:07d}.json"
    with open(file_name, "w") as f:
        json.dump(params, f)

    # shell script
    shell_file = output_dir / f"run_{n:07d}.sh"
    with open(shell_file, 'w') as f:
        f.write(shell_cmds + str(file_name))
    os.chmod(shell_file, 0o775)

    job = {
      "append_output_dir": False,
      "blocked_by": [],
      "command": str(shell_file),
      "extension": "generic_command",
      "job_id": n
    }

    jade_file["jobs"].append(job)

with open(output_dir / "simulate.json", 'w') as f:
    json.dump(jade_file, f)

python_script = """import json
import sys
from pathlib import Path
from pyomo.common.tempfiles import TempfileManager
from dispatches.case_studies.renewables_h2_case.re_h2_parameters import get_gen_outputs_from_rtsgmlc
from dispatches.case_studies.renewables_h2_case.wind_battery_hydrogen_flowsheet import wind_battery_hydrogen_optimize

TempfileManager.tempdir = os.environ.get("LOCAL_SCRATCH")

run_id = params_file.stem.split("_")[1]
result_file = params_file.parent / f"results_{run_id}.json"

with open(params_file, 'r') as f:
    params = json.load(f)

wind_gen = params['wind_gen']
wind_gen_pmax = params['wind_gen_pmax']
gas_gen = params['gas_gen']
reserves = params['reserves']
shortfall = params['shortfall']
start_date = params['start_date']

wind_capacity_factors, loads_mw = get_gen_outputs_from_rtsgmlc(wind_gen, gas_gen, reserves, shortfall, start_date)
params["wind_resource"] = wind_capacity_factors
params["load"] = loads_mw.tolist()

des_res = wind_battery_hydrogen_optimize(n_time_points=8784, input_params=params, verbose=False, plot=False)

params.pop("load")
params.pop("wind_resource")
params.pop("pyo_model")

with open(result_file, 'w') as f:
    json.dump({**params, **des_res}, f)
"""

with open(output_dir / "runner.py", 'w') as f:
    f.write(python_script)

toml_file = """hpc_type = "slurm"
job_prefix = "job"

[hpc]
account = "gmihybridsys"
partition = "short"
walltime = "4:00:00"

#--per-node-batch-size=108 
"""

with open(output_dir / "simulate.toml", 'w') as f:
    f.write(toml_file)