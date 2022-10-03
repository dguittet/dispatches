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
import json
import glob
import sys
from pathlib import Path
import pandas as pd

run_dir = Path(sys.argv[1])

with open(run_dir / "simulate.json", 'r') as f:
    jade_jobs = json.load(f)

completed_runs = []
for res_file in glob.glob(str(run_dir / "result*.json")):
    run_id = int(Path(res_file).stem.split("_")[1])
    completed_runs.append(run_id)

completed_runs.sort()

for run_id in reversed(completed_runs):
    del jade_jobs["jobs"][run_id]

with open(run_dir / "simulate_incomp.json", 'w') as f:
    json.dump(jade_jobs, f)

res_records = []
for res_file in glob.glob(str(run_dir / "result*.json")):
    with open(res_file, 'r') as f:
        res = json.load(f)
    res_records.append(res)

res_df = pd.DataFrame.from_records(res_records)

res_df.to_parquet(run_dir / "aggregated_results.parquet")