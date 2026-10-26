# Session summary — 26 Oct 2026

Thesis project: replicating Ling et al.'s tiny transformer for anomaly detection on the
SKAB dataset (valve1 + valve2, 20 files), first in FP32, then quantized with
ElasticAI.Creator (QAT, integer-only inference, later VHDL for an FPGA).

## 1. What the pipeline does

- **Data** (`data_prep/prepare_data_v2.py`): each of the 20 SKAB valve files is split by row:
  rows 0–319 → training, 320–399 → validation, 400+ → test. Train/validation contain no anomalies.
- **Task** (`core/dataset.py`): one-step-ahead forecasting. Input = the last 24 timesteps of all
  8 sensors (z-scored with training statistics), target = `Volume Flow RateRMS` at the next step.
- **Model** (`core/model.py`): Linear(8→24) → sinusoidal positional encoding → 1 encoder layer
  (single-head attention + FFN, residuals, post-norm **BatchNorm** instead of LayerNorm, for FPGA)
  → mean over time → Linear(24→1). 7465 parameters.
- **Training** (`training/train.py`): MSE, Adam, batch 96, lr 5.064e-3, early stopping (patience 10).
- **Anomaly score** = |target − prediction| (the residual).
- **Evaluation** (`evaluation/evaluate.py`):
  - threshold = 99th percentile of validation residuals;
  - optional "Ling calibration": rescales the threshold per valve group using the median *test*
    residual (so it peeks at test data);
  - per-file F1: a file is a false positive if any alarm before the fault, a true positive if any
    alarm during the fault (20 files → very coarse F1);
  - threshold-free AUC-ROC and AUC-PR per file.

## 2. Changes made today

1. **Fault boundaries.** The old code ended the fault at the 2nd `changepoint`, but SKAB marks
   several changepoints *inside* one anomaly, so only the first ~57 rows of each fault were scored
   (and one file, valve1/2.csv, got ~280). Now the fault = all rows with `anomaly == 1`.
   Ling's version is still available: `--fault-end ling`.
2. **Test context.** Test files now include the 24 rows before row 400 as input only, so the first
   scored prediction is row 400.
3. **Reproducibility.** `--seed` and `--out` arguments; checkpoints no longer overwrite each other.
4. **Validation loss** uses all samples (no `drop_last`), weighted by batch size.
5. **Quantized model** (`core/quantized_model.py`), built from the pasted draft, with fixes:
   - `AVGPooling1d(in_features=window)` — the library divides by `in_features`, so passing
     `d_model` only worked because d_model = window = 24;
   - `precompute()` permanently changes quantizer state (bias bit width 6 → 14), so integer mode
     now runs on a separate copy via `to_integer(model)`;
   - the old self-test passed trivially (see 4.1); the new one trains briefly first;
   - `to_integer()` clones the state: the library's QParams loading *shares* tensors instead of
     copying, so precompute() on the copy silently changed the original's bias/BatchNorm
     quantizers. This made the first `[sim]` numbers slightly wrong (fixed; `[int]` and the
     checkpoints were not affected; verified reloaded == trained, bit-exact).
6. **QAT training** (`training/train_qat.py`): same settings as `train.py`, plus `--bits`;
   saves config + weights; reports integer-mode validation loss at the end.
7. **Folders**: `core/`, `data_prep/`, `training/`, `evaluation/`, `checkpoints/{fp32,qat6,old}/`,
   `figures/`. Scripts are run as modules from the project root (see section 6).

## 3. Results (5 seeds each, mean ± std)

**Full labelled fault (default):**

| Model | F1 calibrated | F1 fixed | AUC-ROC | AUC-PR |
|---|---|---|---|---|
| FP32 | 0.871 ± 0.035 | 0.709 ± 0.006 | 0.723 ± 0.052 | 0.838 (random 0.696) |
| QAT 6-bit, simulated | 0.819 ± 0.009 | 0.730 ± 0.020 | 0.831 ± 0.024 | 0.903 |
| QAT 6-bit, integer | 0.824 ± 0.000 | 0.742 ± 0.025 | 0.831 ± 0.027 | 0.907 |

**Ling's fault cut (comparable to the paper's F1 = 0.766):**

| Model | F1 calibrated | F1 fixed | AUC-ROC |
|---|---|---|---|
| FP32 | 0.688 ± 0.053 | 0.637 ± 0.020 | 0.670 |
| QAT 6-bit, integer | 0.724 ± 0.008 | 0.662 ± 0.029 | 0.681 |

Validation MSE (z-scored target): FP32 ≈ 0.77, integer 6-bit ≈ 0.77–0.79.

## 4. Caveats and findings (important for the thesis)

1. **BatchNorm init collapse.** At init, BatchNorm's γ is all 1.0, the min/max observer sees
   range [1, 1] and picks scale ≈ 1e-7, so the encoder outputs zeros. Training recovers because
   the observer keeps the widest range seen, but any test on an untrained model is meaningless.
2. **F1 is lenient.** A detector that always alarms gets F1 = 0.667 under this protocol.
   Uncalibrated F1 (~0.71) is only slightly above that; most files get a false alarm.
3. **The calibration uses test data** (median test residual, fault included).
4. **The forecaster is weak.** Validation MSE 0.77 vs 1.0 for predicting a constant (~23% of
   variance explained). A model-free score (sum of |z-scores| of the current readings) reaches
   AUC-ROC 0.80, higher than the FP32 model (0.72).
5. **Replication:** under the paper's protocol FP32 gets 0.688 ± 0.053 (seeds range 0.64–0.78);
   the paper's 0.766 lies inside that range, likely a single run.
6. **Quantization costs nothing here.** 6-bit integer inference matches FP32 on F1.
7. **Why the quantized AUC-ROC is higher (0.83 vs 0.72): output saturation, not better learning.**
   The integer model can only output values inside its output quantizer range (learned on normal
   data, z ∈ [−2.11, 4.90]); 76% of fault-time targets lie outside it. FP32 follows the fault
   (flow rate is in its input) and keeps residuals small; the integer model cannot, so residuals
   grow. Clipping FP32's output to the same range gives AUC 0.817 — nearly all of the gain.
   On the early fault (Ling's cut) there is no gain. Do not present it as a quantization benefit.
8. **Simulated vs integer outputs are not bit-exact** (up to ~7–9 quant steps apart; rounding in
   multiply-and-shift and the softmax lookup table). Report integer-mode numbers: that is what the
   FPGA computes.
9. **Architecture differences FP32 vs QAT:** the library attention has no dropout (FP32 uses 0.2);
   the QAT model is trained from scratch, not initialized from FP32 weights.

## 5. Open questions / next steps

1. Commit the current state (nothing is committed yet).
2. Bit-width sweep: 4 and 8 bits, 5 seeds each — where does accuracy start to drop?
3. Generate VHDL from a trained QAT model with ElasticAI.Creator.
4. Possibly add the trivial baselines (always-alarm F1, z-score AUC) to `evaluate.py`.
5. Write up results and caveats.

## 6. Commands (run from the project root)

```fish
.venv/bin/python -m data_prep.prepare_data_v2                 # rebuild data/
.venv/bin/python -m training.train --seed 0                    # -> checkpoints/fp32/seed0.pt
.venv/bin/python -m training.train_qat --seed 0 --bits 6       # -> checkpoints/qat6/seed0.pt
.venv/bin/python -m evaluation.evaluate checkpoints/fp32/*.pt checkpoints/qat6/*.pt
.venv/bin/python -m evaluation.evaluate --fault-end ling checkpoints/qat6/*.pt
```

## 7. Issues found in Ling et al.'s work

Caveat: their full repository was not reviewed. Items 1–2 come from the original replication code,
which was written to follow their `test.py` / `uva_evaluate`; verify them against their code
before stating them in the thesis. Items 6–7 come from their `QuantTransformer` (pasted and compared).

**Evaluation protocol**
1. **Fault cut short.** The fault end is taken as the 2nd `changepoint`, but SKAB marks several
   changepoints inside one fault, so only the first ~57 of ~400 fault rows are scored
   (valve1/2.csv, with 3 changepoints, keeps 280). The `anomaly` column defines the fault.
2. **Threshold calibration uses test data** (median residual of the test files, fault included).
3. **Lenient F1.** One alarm anywhere in the fault counts as a detection; an always-alarm detector
   scores F1 = 0.667, and the paper does not compare against that.
4. **Single run reported.** Replication over 5 seeds: F1 = 0.688 ± 0.053 (range 0.64–0.78);
   the paper's 0.766 is at the top of that range.
5. **No simple baseline.** A model-free score (sum of |z-scores|) reaches AUC-ROC 0.80,
   above the transformer's 0.72.

**Quantized model code**
6. **Integer mode not stable across calls.** `precompute()` runs before every integer-mode call and
   changes quantizer state, so the first call differs from later ones (6–30% of outputs, by 1–2
   quant steps). The generated VHDL depends on when `create_design()` runs.
7. **Pooling wired correctly by coincidence.** `AVGPooling1d(in_features=d_model)`: the library
   divides by `in_features` as the window length; correct only because d_model = window = 24.

**Not Ling's errors, but worth mentioning**
- Library (ElasticAI.Creator): QParams state loading shares tensors instead of copying them;
  BatchNorm γ = 1 at init gives a degenerate quantizer range (encoder outputs 0 until γ moves).
- Dataset: the flow rate is mostly unpredictable second to second. Gradient boosting reaches MSE
  0.23 only by using decimal artifacts of the logger (e.g. 32.9962 → 32.0040 → 32.0000); with values
  rounded it gets 1.00. The transformer (0.77) is about as good as linear regression (0.78), i.e.
  near the realistic limit.

Items 1–5 affect how the reported numbers should be interpreted, not whether the method works.
Items 6–7 are small code issues that do not change results at the paper's settings.
