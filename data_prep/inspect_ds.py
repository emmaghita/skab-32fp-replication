import pandas as pd
import matplotlib.pyplot as plt

SRC = "SKAB-master/data"
COLS = ["Volume Flow RateRMS", "Accelerometer1RMS", "Temperature"]

files = {
    "anomaly-free (training)": f"{SRC}/anomaly-free/anomaly-free.csv",
    "valve1/0.csv (test)":     f"{SRC}/valve1/0.csv",
    "valve2/0.csv (test)":     f"{SRC}/valve2/0.csv",
}

# 1) numbers: mean of each sensor per file
table = {name: pd.read_csv(path, sep=";")[COLS].mean() for name, path in files.items()}
print(pd.DataFrame(table).round(3))

# 2) plot: each sensor over time, all three files on the same axes
fig, axes = plt.subplots(len(COLS), 1, figsize=(10, 8), sharex=True)
for name, path in files.items():
    df = pd.read_csv(path, sep=";")
    for ax, col in zip(axes, COLS):
        ax.plot(df[col].values[:1200], label=name)
        ax.set_ylabel(col)
axes[0].legend()
axes[-1].set_xlabel("time step (seconds)")
plt.tight_layout()
plt.savefig("figures/regimes.png", dpi=120)
plt.show()