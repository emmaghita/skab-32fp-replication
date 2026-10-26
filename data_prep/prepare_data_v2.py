import os, glob
import pandas as pd

from core.dataset import WINDOW

src = "../SKAB/data"
out = "data/SKABData_v2"
for d in ["normal", "valid", "valve1", "valve2"]:
    os.makedirs(f"{out}/{d}", exist_ok=True)

for group in ["valve1", "valve2"]:
    for path in sorted(glob.glob(f"{src}/{group}/*.csv")):
        df = pd.read_csv(path, sep=";")
        name = f"{group}_{os.path.basename(path)}"
        df.iloc[:320].to_csv(f"{out}/normal/{name}", sep=";", index=False)      # train: rows 0-319
        df.iloc[320:400].to_csv(f"{out}/valid/{name}", sep=";", index=False)    # valid: rows 320-399
        # test: 400+, plus the WINDOW rows before it as context only, so the first scored
        # prediction is row 400 itself (those context rows are never targets)
        df.iloc[400 - WINDOW:].to_csv(f"{out}/{group}/{os.path.basename(path)}", sep=";", index=False)

print("done")
