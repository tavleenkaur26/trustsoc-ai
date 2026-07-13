"""
Build a diverse ~40-row CSV sampled from ALL raw parquet files (proportional
across classes), to be uploaded through predict_pipeline.py + the dashboard's
real model + SHAP, then batch-run through llm_report.py for hallucination
evaluation on genuinely real, diverse data.
"""
import pandas as pd
import glob

files = glob.glob("data/raw/*.parquet")
samples = []
per_file = 6  # 8 files * 6 = ~48 rows, spanning all classes after merge

for f in files:
    df = pd.read_parquet(f).drop(columns=["Label"], errors="ignore")
    n = min(per_file, len(df))
    samples.append(df.sample(n, random_state=7))

out = pd.concat(samples, ignore_index=True)
out.to_csv("real_eval_batch.csv", index=False)
print(f"Saved real_eval_batch.csv with {len(out)} rows from {len(files)} source files")