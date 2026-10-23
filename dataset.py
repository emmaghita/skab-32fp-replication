import glob
import numpy as np
import pandas as pd

FEATURES = ["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
            "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS"]
WINDOW = 24
DATA = "data/SKABData_v2";

def read_file(path):
    return pd.read_csv(path, sep=";")

def make_windows(x):
    X, y = [], []
    for t in range(WINDOW, len(x)):
        X.append(x[t - WINDOW : t]) #24 rows before t
        y.append([x[t, -1]]) #flow rate at row t
    return np.stack(X), np.array(y, dtype=np.float32)

def load_split(name, mean, std, test=False):
    X_all, y_all, fault_starts = [], [], []
    for path in sorted(glob.glob(f"{DATA}/{name}/*.csv")):
        df = read_file(path)
        if test:
            cps = df.index[df["changepoint"] == 1].tolist()
            df = df.iloc[:cps[1]] 
            fault_starts.append(cps[0] - WINDOW)
        x = (df[FEATURES].to_numpy(np.float32) - mean) / std
        X, y = make_windows(x)
        X_all.append(X)
        y_all.append(y)
    return X_all, y_all, fault_starts 

def get_data():
    normal = np.vstack([read_file(p)[FEATURES].to_numpy(np.float32)
                        for p in sorted(glob.glob(f"{DATA}/normal/*.csv"))])
    mean, std = normal.mean(axis=0), normal.std(axis=0)
    train = load_split("normal", mean, std)
    valid = load_split("valid", mean, std)
    test = [load_split("valve1", mean, std, test=True),
            load_split("valve2", mean, std, test=True)]
    return train, valid, test, mean, std

if __name__ == "__main__":
    train, valid, test, mean, std = get_data()
    print("train:", train[0][0].shape, train[1][0].shape)
    print("valid:", valid[0][0].shape)
    print("valve1 files:", len(test[0][0]), "| first fault start:", test[0][2][0])
    print("valve2 files:", len(test[1][0]))