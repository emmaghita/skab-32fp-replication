import numpy as np
import torch

from sklearn.metrics import roc_auc_score, average_precision_score

from dataset import get_data
from model import TinyTransformer

D_MODEL = 24          # must match train.py
ALPHA = 0.279         # Ling's calibration constant (their test.py, line 323)


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

if __name__ == "__main__":
    train, valid, test, mean, std = get_data()

    model = TinyTransformer(num_in=8, num_out=1, d_model=D_MODEL)
    model.load_state_dict(torch.load("best_model.pt", weights_only=True))
    model.eval()                                       # dropout off, BatchNorm uses learned stats

    # step 1: threshold from validation data
    val_res = residuals_of(model, np.concatenate(valid[0]), np.concatenate(valid[1]))
    threshold = np.percentile(val_res, 99)
    beta = np.median(val_res)
    print(f"validation: threshold (99th pct) {threshold:.4f}, median {beta:.4f}")

    # step 2: test
    print("\nWith Ling's calibration (compare with paper, F1 = 0.766):")
    evaluate(model, test, threshold, beta, calibrate=True)

    print("\nWithout calibration (fixed validation threshold):")
    evaluate(model, test, threshold, beta, calibrate=False)

    print("\nThreshold-free metrics (per file, averaged over 20 files):")
    auc_scores(model, test)