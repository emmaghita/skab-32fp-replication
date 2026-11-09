# Anomaly detection computed from the SIMULATED HARDWARE outputs of the full validation + test set.
#
# Input: <design>/full/ made by `convert.py --full` and simulated by `sim.sh <design> --full`:
#   shard_XX/data/transformer_q_y.txt    expected integers (Python integer model)
#   shard_XX/data/transformer_q_out.txt  integers computed by the VHDL (Vivado simulation)
#
# Output:
#   1) how many windows the VHDL reproduces exactly (should be all of them)
#   2) F1 fixed from the hardware outputs, with the same protocol as evaluation/evaluate.py:
#      threshold = 99th percentile of the validation scores, per-file F1 on the 20 test files,
#      full labelled fault. Also the same F1 from the Python integer outputs, for comparison.
#
# Run from software/:   uv run python -m hw_design.hw_eval ../hardware/qat6_seed1

import json
import sys
from pathlib import Path

import numpy as np
import torch

from core.dataset import get_data
from core.quantized_model import to_integer
from evaluation.evaluate import load_models


def read_ints(path):
    return np.loadtxt(path, dtype=int, ndmin=1) if path.exists() else None


def f1_from_scores(val_scores, test_scores, test_files):
    # same rule as evaluation/evaluate.py: one fixed threshold, per-file TP / FP / FN
    thr = np.percentile(val_scores, 99)
    tp = fp = fn = 0
    start = 0
    for n, fs in test_files:                              # (windows in this file, fault start)
        s = test_scores[start:start + n]
        start += n
        fp += (s[:fs] >= thr).any()
        tp += (s[fs:] >= thr).any()
        fn += not (s[fs:] >= thr).any()
    pre = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0.0
    return f1, thr, tp, fp, fn


full = Path(sys.argv[1]) / "full"
man = json.load(open(full / "manifest.json"))
shards = [full / f"shard_{k:02d}" / "data" for k in range(len(man["shards"]))]

expected, hardware, missing = [], [], []
for k, (d, n) in enumerate(zip(shards, man["shards"])):
    e, h = read_ints(d / f"{man['name']}_q_y.txt"), read_ints(d / f"{man['name']}_q_out.txt")
    if h is None or len(h) != n:
        missing.append(k)
        continue
    expected.append(e)
    hardware.append(h)

if missing:
    print(f"shards not (completely) simulated: {missing} -> run ../hardware/sim.sh <design> --full")
    sys.exit(1)
expected, hardware = np.concatenate(expected), np.concatenate(hardware)

# 1) exact match: hardware integers == Python integers
same = expected == hardware
n_val = min(man["n_valid"], len(hardware))
print(f"windows simulated  : {len(hardware)}  ({n_val} validation + {len(hardware) - n_val} test)")
print(f"identical to Python: {same.sum()} / {len(same)}" + ("" if same.all() else
      f"  -> {np.flatnonzero(~same)[:10].tolist()} ... differ"))

if man["limit"] is not None:
    print("(--limit was used: not the full set, no F1)")
    sys.exit(0 if same.all() else 1)

# 2) integers -> flow values with the model's own dequantize (exactly as int_forward does)
model = to_integer(load_models(man["checkpoint"])[0][1])
deq = lambda q: model.output_linear.outputs_QParams.dequantize(torch.tensor(q, dtype=torch.int32)).numpy()

_, valid, test, *_ = get_data()                           # same order as convert.py --full
y_true = np.concatenate(list(valid[1]) + [y for _, y_files, _ in test for y in y_files])[:, 0]
test_files = [(len(X), fs) for X_files, _, fss in test for X, fs in zip(X_files, fss)]
nv = man["n_valid"]

for label, q in (("hardware (VHDL)", hardware), ("Python integer", expected)):
    score = np.abs(y_true - deq(q))                       # anomaly score = |real - predicted|
    f1, thr, tp, fp, fn = f1_from_scores(score[:nv], score[nv:], test_files)
    print(f"F1 fixed, {label:<16}: {f1:.3f}   (threshold {thr:.3f}, TP {tp}, FP {fp}, FN {fn})")

sys.exit(0 if same.all() else 1)
