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