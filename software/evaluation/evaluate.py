import os
import numpy as np
import torch
import argparse

from sklearn.metrics import roc_auc_score, average_precision_score

from core.dataset import get_data
from core.model import TinyTransformer
from core.quantized_model import QuantTinyTransformer, to_integer

D_MODEL = 24          # must match train.py
ALPHA = 0.279         # Ling's calibration constant (their test.py, line 323)


def load_models(path):
    # FP32 checkpoint (state_dict) -> one model; QAT checkpoint -> simulated + integer model
    ckpt = torch.load(path, weights_only=True)
    if "config" not in ckpt:
        model = TinyTransformer(num_in=8, num_out=1, d_model=D_MODEL)
        model.load_state_dict(ckpt)
        return [(path, model.eval())]
    model = QuantTinyTransformer(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    try:
        return [(f"{path} [sim]", model), (f"{path} [int]", to_integer(model))]
    except ZeroDivisionError:   # degenerate quantizer range (e.g. 3-bit models that never learned)
        print(f"NOTE: {path}: integer conversion failed (degenerate quantizer range), [sim] only")
        return [(f"{path} [sim]", model)]


def predict(model, X):
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy()


def residuals_of(model, X, y):
    return np.abs(y - predict(model, X)).sum(axis=1)       # |target - prediction|


def evaluate(model, test, threshold, beta=None, calibrate=False):
    # per-file F1 (Ling's uva_evaluate). calibrate=True: Ling's calibration, which uses the
    # median test residual (test data), kept only for comparing with the paper
    tp = fp = fn = 0
    for X_files, y_files, fault_starts in test:                # valve1, valve2
        res_files = [residuals_of(model, X, y) for X, y in zip(X_files, y_files)]
        thr = threshold
        if calibrate:
            scale = np.median(np.concatenate(res_files)) / beta
            thr = threshold * (1 - ALPHA + ALPHA * scale)
        for res, fs in zip(res_files, fault_starts):
            if (res[:fs] >= thr).any(): fp += 1     # alarm before the fault -> false positive
            if (res[fs:] >= thr).any(): tp += 1     # alarm during the fault -> detected
            else:                       fn += 1
    pre = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * pre * rec / (pre + rec) if pre + rec else 0.0


def auc_scores(model, test):
    # threshold-free metrics per file, averaged (not printed by default)
    aucs, aps = [], []
    for X_files, y_files, fault_starts in test:
        for X, y, fs in zip(X_files, y_files, fault_starts):
            res = residuals_of(model, X, y)
            labels = np.zeros(len(res)); labels[fs:] = 1
            aucs.append(roc_auc_score(labels, res))
            aps.append(average_precision_score(labels, res))
    return np.mean(aucs), np.mean(aps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=["checkpoints/fp32/seed0.pt"])
    parser.add_argument("--fault-end", choices=["anomaly", "ling"], default="anomaly",
                        help="anomaly: full labelled fault; ling: cut at 2nd changepoint like Ling's code")
    args = parser.parse_args()

    train, valid, test, mean, std = get_data(args.fault_end)

    # F1 fixed: threshold = 99th percentile of the model's own validation residuals,
    # no calibration, no test data -> what the FPGA would use
    rows = []
    print(f"{'model':<40}{'F1 fixed':>10}")
    for name, model in [m for path in args.paths for m in load_models(path)]:
        val_res = residuals_of(model, np.concatenate(valid[0]), np.concatenate(valid[1]))
        f1 = evaluate(model, test, np.percentile(val_res, 99))
        rows.append((name, f1))
        print(f"{name:<40}{f1:>10.3f}")

    # mean ± std per group: checkpoint folder + variant, e.g. "qat6 [int]"
    def group_of(name):
        path, _, variant = name.partition(" ")
        return f"{os.path.basename(os.path.dirname(path))} {variant}".strip()
    groups = dict.fromkeys(group_of(n) for n, _ in rows)
    if any(sum(group_of(n) == g for n, _ in rows) > 1 for g in groups):
        print(f"\n{'group (mean ± std over seeds)':<40}{'F1 fixed':>16}")
        for g in groups:
            f = np.array([f1 for n, f1 in rows if group_of(n) == g])
            print(f"{f'{g} (n={len(f)})':<40}{f.mean():>9.3f} ± {f.std():.3f}")
