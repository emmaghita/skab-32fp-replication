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
    return [(f"{path} [sim]", model), (f"{path} [int]", to_integer(model))]


def predict(model, X):
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy()


def residuals_of(model, X, y):
    return np.abs(y - predict(model, X)).sum(axis=1)       # |target - prediction|


def evaluate(model, test, threshold, beta, calibrate):
    tp = fp = fn = tn = 0
    for name, (X_files, y_files, fault_starts) in zip(["valve1", "valve2"], test):
        # residuals for every file in this group
        res_files = [residuals_of(model, X, y) for X, y in zip(X_files, y_files)]

        # Ling's calibration: one threshold per group, scaled by the group's median residual
        thr = threshold
        if calibrate:
            scale = np.median(np.concatenate(res_files)) / beta
            thr = threshold * (1 - ALPHA + ALPHA * scale)

        # per-file scoring (Ling's uva_evaluate)
        for res, fs in zip(res_files, fault_starts):
            if (res[:fs] >= thr).any(): fp += 1     # alarm before the fault -> false positive
            else:                       tn += 1
            if (res[fs:] >= thr).any(): tp += 1     # alarm during the fault -> detected
            else:                       fn += 1
        print(f"  {name}: threshold {thr:.4f}")

    pre = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0.0
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn} | precision {pre:.3f} recall {rec:.3f} F1 {f1:.3f}")
    return f1

def auc_scores(model, test):
    aucs, aps, base = [], [], []
    for X_files, y_files, fault_starts in test:                # valve1, valve2
        for X, y, fs in zip(X_files, y_files, fault_starts):   # each file
            res = residuals_of(model, X, y)                    # score = residual
            labels = np.zeros(len(res))
            labels[fs:] = 1                                    # 0 = normal, 1 = fault
            aucs.append(roc_auc_score(labels, res))
            aps.append(average_precision_score(labels, res))   # = AUC-PR
            base.append(labels.mean())                         # AUC-PR of a random guess
    print(f"  AUC-ROC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  (random = 0.5)")
    print(f"  AUC-PR:  {np.mean(aps):.3f} ± {np.std(aps):.3f}  (random = {np.mean(base):.3f})")
    return np.mean(aucs), np.mean(aps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=["checkpoints/fp32/seed0.pt"])
    parser.add_argument("--fault-end", choices=["anomaly", "ling"], default="anomaly",
                        help="anomaly: full labelled fault; ling: cut at 2nd changepoint like Ling's code")
    args = parser.parse_args()

    train, valid, test, mean, std = get_data(args.fault_end)
    paths = args.paths
    rows = []

    for name, model in [m for path in paths for m in load_models(path)]:
        print(f"\n===== {name} =====")

        val_res = residuals_of(model, np.concatenate(valid[0]), np.concatenate(valid[1]))
        threshold = np.percentile(val_res, 99)
        beta = np.median(val_res)
        print(f"validation: threshold (99th pct) {threshold:.4f}, median {beta:.4f}")

        print("\nWith Ling's calibration (compare with paper, F1 = 0.766):")
        f1_cal = evaluate(model, test, threshold, beta, calibrate=True)

        print("\nWithout calibration (fixed validation threshold):")
        f1_fix = evaluate(model, test, threshold, beta, calibrate=False)

        print("\nThreshold-free metrics (per file, averaged over 20 files):")
        roc, pr = auc_scores(model, test)
        rows.append((name, f1_cal, f1_fix, roc, pr))

    print("\n===== SUMMARY =====")
    print(f"{'model':<34}{'F1 cal':>8}{'F1 fixed':>10}{'AUC-ROC':>9}{'AUC-PR':>8}")
    for r in rows:
        print(f"{r[0]:<34}{r[1]:>8.3f}{r[2]:>10.3f}{r[3]:>9.3f}{r[4]:>8.3f}")
    # mean/std per variant (fp32 / [sim] / [int]), so the variants are not mixed
    for kind in dict.fromkeys(r[0].split(" ")[-1] if r[0].endswith("]") else "fp32" for r in rows):
        group = [r for r in rows if (r[0].split(" ")[-1] if r[0].endswith("]") else "fp32") == kind]
        if len(group) > 1:
            m = np.array([r[1:] for r in group])
            print(f"{'mean ' + kind:<34}" + "".join(f"{v:>{w}.3f}" for v, w in zip(m.mean(0), [8, 10, 9, 8])))
            print(f"{'std ' + kind:<34}" + "".join(f"{v:>{w}.3f}" for v, w in zip(m.std(0), [8, 10, 9, 8])))
