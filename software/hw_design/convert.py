# Trained QAT checkpoint -> everything Vivado needs (my version of Ling's hw_converter/convert2hw.py).
#
#   <out>/data/     integer test inputs + expected outputs for the testbenches
#                   (transformer_q_x.txt, transformer_q_y.txt, plus one pair per layer)
#   <out>/source/   the VHDL: 5 layers, top level transformer.vhd, PE ROM, testbenches
#
# Run from software/:   uv run python -m hw_design.convert
# Then simulate from inside <out>, because the testbench reads ./data/...
#
# Data and VHDL come from the same integer model (one to_integer() call), so the
# testbench compares the hardware with exactly the model that was evaluated.

import argparse
import shutil
from pathlib import Path

import torch
from elasticai.creator.file_generation.on_disk_path import OnDiskPath

from core.dataset import get_data
from core.quantized_model import to_integer
from evaluation.evaluate import load_models

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="checkpoints/qat6/seed1.pt",
                    help="QAT checkpoint (seed 1 = lowest integer validation loss)")
parser.add_argument("--out", default=None, help="output folder (default: ../hardware/<qatN>_<seedM>)")
parser.add_argument("--group", type=int, default=0, help="test group: 0 = valve1, 1 = valve2")
parser.add_argument("--file", type=int, default=0, help="test file inside the group")
parser.add_argument("--windows", type=int, default=10, help="test windows, centred on the fault start")
args = parser.parse_args()

ckpt = Path(args.ckpt)
out = Path(args.out or f"../hardware/{ckpt.parent.name}_{ckpt.stem}")     # e.g. ../hardware/qat6_seed1
for sub in ("data", "source"):                    # only the generated folders are replaced
    shutil.rmtree(out / sub, ignore_errors=True)
(out / "data").mkdir(parents=True)

# 1) trained QAT model -> integer copy that saves its integer data into <out>/data
model = load_models(str(ckpt))[0][1]              # [0] = the [sim] model as trained
int_model = to_integer(model, quant_data_dir=str(out / "data"))

# 2) real test windows: half before the fault, half during it.
#    One call only: every call overwrites the data files.
_, _, test, *_ = get_data()
X_files, _, fault_starts = test[args.group]
fs, half = fault_starts[args.file], args.windows // 2
X = torch.from_numpy(X_files[args.file][fs - half : fs + args.windows - half])
with torch.no_grad():
    y = int_model(X)

# 3) VHDL from the same integer model
int_model.create_design(int_model.name).save_to(OnDiskPath(name="source", parent=str(out)))

n_vhd = len(list((out / "source").rglob("*.vhd")))
print(f"checkpoint : {ckpt}")
print(f"test data  : {len(X)} windows of {['valve1', 'valve2'][args.group]}/{args.file} "
      f"({half} normal, {len(X) - half} fault) -> {out / 'data'}")
print(f"VHDL       : {n_vhd} files -> {out / 'source'}")
print(f"top level  : {out / 'source' / int_model.name / (int_model.name + '.vhd')}")
print(f"testbench  : {out / 'source' / int_model.name / (int_model.name + '_tb.vhd')}")
print(f"predicted flow (z-scored): {[round(v, 3) for v in y.flatten().tolist()]}")
