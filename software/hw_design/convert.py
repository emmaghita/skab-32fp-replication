# Trained QAT checkpoint -> everything Vivado needs (my version of Ling's hw_converter/convert2hw.py).
#
#   <out>/data/     integer test inputs + expected outputs for the testbenches
#                   (transformer_q_x.txt, transformer_q_y.txt, plus one pair per layer)
#   <out>/source/   the VHDL: 5 layers, top level transformer.vhd, PE ROM, testbenches
#   <out>/full/     only with --full: ALL validation + test windows, split into shards
#                   (shard_XX/data/transformer_q_x.txt + _q_y.txt) to simulate in parallel,
#                   plus manifest.json (what is in which shard) for hw_eval.py
#
# Run from software/:   uv run python -m hw_design.convert [--full]
# Then: ../hardware/sim.sh [<out name>] [--full]
#
# Data and VHDL come from the same integer model (one to_integer() call), so the
# testbench compares the hardware with exactly the model that was evaluated.

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from elasticai.creator.file_generation.on_disk_path import OnDiskPath

from core.dataset import get_data
from core.quantized_model import to_integer
from evaluation.evaluate import load_models

# --- make the generated VHDL saturate exactly like the Python integer model ---
# Python (ElasticAI's clamp_result) clamps every layer output to 6 bits, but the VHDL
# templates cut off the top bits with resize() (wrap-around: 35 -> 3 instead of 31).
# Found with the full-set simulation: 4503 / 12359 windows differed. Two fixes:
#  1) scaling(): clamp the rescaled magnitude to 2**DATA_WIDTH - 1 before the 7-bit resize
#  2) output store: add Z_Y as an integer, then clamp to [-2**(DATA_WIDTH-1), 2**(DATA_WIDTH-1) - 1]
# Exact (not approximate): every Z_Y is in [-32, 31], so the clamp at 63 never changes the result.
SCALE = re.compile(r"(\n\s*)(if is_negative then\s*return -resize\(TMP_3, DATA_WIDTH \+ 1\);)")
SCALE_NEW = r"\1if TMP_3 > 2**DATA_WIDTH - 1 then TMP_3 := to_signed(2**DATA_WIDTH - 1, TMP_3'length); end if;\1\2"
STORE = re.compile(r"var_y_store := (\w+) \+ to_signed\(Z_Y, \w+'length\);(\s*)"
                   r"y_store_data <= std_logic_vector\(resize\(var_y_store, y_store_data'length\)\);")
STORE_NEW = (r"if to_integer(\1) + Z_Y > 2**(DATA_WIDTH-1) - 1 then"
             r"\2    y_store_data <= std_logic_vector(to_signed(2**(DATA_WIDTH-1) - 1, y_store_data'length));"
             r"\2elsif to_integer(\1) + Z_Y < -(2**(DATA_WIDTH-1)) then"
             r"\2    y_store_data <= std_logic_vector(to_signed(-(2**(DATA_WIDTH-1)), y_store_data'length));"
             r"\2else"
             r"\2    y_store_data <= std_logic_vector(to_signed(to_integer(\1) + Z_Y, y_store_data'length));"
             r"\2end if;")


def saturate_like_python(source):
    n_scale = n_store = 0
    for f in Path(source).rglob("*.vhd"):
        text = f.read_text()
        text, a = SCALE.subn(SCALE_NEW, text)
        text, b = STORE.subn(STORE_NEW, text)
        if a or b:
            f.write_text(text)
        n_scale, n_store = n_scale + a, n_store + b
    return n_scale, n_store


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="checkpoints/qat6/seed1.pt",
                    help="QAT checkpoint (seed 1 = lowest integer validation loss)")
parser.add_argument("--out", default=None, help="output folder (default: ../hardware/<qatN>_<seedM>)")
parser.add_argument("--group", type=int, default=0, help="test group: 0 = valve1, 1 = valve2")
parser.add_argument("--file", type=int, default=0, help="test file inside the group")
parser.add_argument("--windows", type=int, default=10, help="test windows, centred on the fault start")
parser.add_argument("--full", action="store_true",
                    help="also write ALL validation + test windows, split into shards (for hw_eval.py)")
parser.add_argument("--shards", type=int, default=os.cpu_count(), help="parallel parts for --full")
parser.add_argument("--limit", type=int, default=None, help="--full with only the first N windows (quick check)")
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

# 3) VHDL from the same integer model, patched to saturate like Python (see saturate_like_python)
int_model.create_design(int_model.name).save_to(OnDiskPath(name="source", parent=str(out)))
n_scale, n_store = saturate_like_python(out / "source")
assert (n_scale, n_store) == (13, 16), f"saturation patch: {n_scale} scaling, {n_store} stores (expected 13, 16)"

# 4) --full: every validation and test window, in a fixed order, split into shards.
#    Only the model's own input/output files are saved here (the per-layer files would be
#    gigabytes for 12k windows): the layers keep quant_data_dir=None, and only the model's
#    top-level quant_data_dir is set, once per shard. Same integer model as above.
shutil.rmtree(out / "full", ignore_errors=True)
if args.full:
    _, valid, test, *_ = get_data()                       # full labelled fault, as in evaluate.py
    files = list(valid[0]) + [X for X_files, _, _ in test for X in X_files]   # valid, valve1, valve2
    X_all = np.concatenate(files)[: args.limit]
    full_model = to_integer(model)                        # layers: quant_data_dir = None
    sizes = []
    for k, chunk in enumerate(np.array_split(X_all, args.shards)):
        full_model.quant_data_dir = str(out / "full" / f"shard_{k:02d}" / "data")
        os.makedirs(full_model.quant_data_dir)
        with torch.no_grad():
            full_model(torch.from_numpy(chunk))           # writes transformer_q_x / _q_y.txt
        sizes.append(len(chunk))
    json.dump({"checkpoint": str(ckpt), "name": full_model.name, "shards": sizes,
               "n_valid": sum(len(X) for X in valid[0]), "limit": args.limit},
              open(out / "full" / "manifest.json", "w"), indent=1)

n_vhd = len(list((out / "source").rglob("*.vhd")))
print(f"checkpoint : {ckpt}")
print(f"test data  : {len(X)} windows of {['valve1', 'valve2'][args.group]}/{args.file} "
      f"({half} normal, {len(X) - half} fault) -> {out / 'data'}")
print(f"VHDL       : {n_vhd} files -> {out / 'source'}  (saturation patch: {n_scale} scaling, {n_store} stores)")
print(f"top level  : {out / 'source' / int_model.name / (int_model.name + '.vhd')}")
print(f"testbench  : {out / 'source' / int_model.name / (int_model.name + '_tb.vhd')}")
print(f"predicted flow (z-scored): {[round(v, 3) for v in y.flatten().tolist()]}")
if args.full:
    print(f"full set   : {sum(sizes)} windows (validation + test) in {len(sizes)} shards -> {out / 'full'}")
