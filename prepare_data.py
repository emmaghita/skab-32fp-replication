import os, shutil
import pandas as pd

src = "SKAB-master/data"
out = "data/SKABData"

# split the anomaly-free 80% t, 20% split
df = pd.read_csv(f"{src}/anomaly-free/anomaly-free.csv", sep=";")
cut = int(len(df) * 0.8)
os.makedirs(f"{out}/normal", exist_ok=True)
os.makedirs(f"{out}/valid", exist_ok=True)
df.iloc[:cut].to_csv(f"{out}/normal/normal.csv", sep=";", index=False)
df.iloc[cut:].to_csv(f"{out}/valid/valid.csv", sep=";", index=False)

# test data: copy as-is
shutil.copytree(f"{src}/valve1", f"{out}/valve1", dirs_exist_ok=True)
shutil.copytree(f"{src}/valve2", f"{out}/valve2", dirs_exist_ok=True)

print("train rows:", cut, "| valid rows:", len(df) - cut)