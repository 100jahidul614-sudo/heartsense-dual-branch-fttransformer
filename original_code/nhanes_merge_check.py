import os
import pandas as pd
from functools import reduce

# 1. Set your NHANES folder path
DATA_DIR = r"C:\Users\<user>\Research center\Heartseanse2.0\Dataset2"  # change this

# 2. List files to merge
files = [
    "P_DEMO.xpt",
    "P_BMX.xpt",
    "P_BPXO.xpt",
    "P_BPQ.xpt",
    "P_DIQ.xpt",
    "P_MCQ.xpt",
    "P_PAQ.xpt",
    "P_SMQ.xpt",
    "P_ALB_CR.xpt",
    "P_BIOPRO.xpt",
    "P_CBC.xpt",
    "P_GHB.xpt",
    "P_GLU.xpt",
    "P_HDL.xpt",
    "P_HSCRP.xpt",
    "P_INS.xpt",
    "P_TCHOL.xpt",
    "P_TRIGLY.xpt",
]

# 3. Load each table
dfs = []

for file in files:
    path = os.path.join(DATA_DIR, file)
    df = pd.read_sas(path, format="xport")
    print(f"{file}: {df.shape}")
    dfs.append(df)

# 4. Merge all tables using SEQN
merged = reduce(
    lambda left, right: pd.merge(left, right, on="SEQN", how="left"),
    dfs
)

print("\nMerged dataset shape:", merged.shape)

# 5. Save merged CSV
output_path = os.path.join(DATA_DIR, "nhanes_2017_2020_merged_raw.csv")
merged.to_csv(output_path, index=False)

print("Saved:", output_path)