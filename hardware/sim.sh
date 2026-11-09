#!/usr/bin/env bash
# Simulate a generated design in Vivado's simulator and check it against the Python integer model.
#
#   ./hardware/sim.sh                     # default design folder: hardware/qat6_seed1
#   ./hardware/sim.sh qat6_seed0          # another folder made by convert.py
#   ./hardware/sim.sh qat6_seed1 --full   # ALL validation + test windows (convert.py --full),
#                                         # in parallel, then F1 from the hardware outputs
#
# Steps: 1 compile (xvhdl) -> 2 elaborate (xelab) -> 3 run the testbench (xsim) -> 4 compare.
# PASS = the VHDL outputs (data/transformer_q_out.txt) equal the Python integer outputs
# (data/transformer_q_y.txt) for every test window.
set -euo pipefail

VIVADO_BIN="${VIVADO_BIN:-$HOME/Vivado/2025.2/Vivado/bin}"   # override: VIVADO_BIN=... ./sim.sh
HW="$(cd "$(dirname "$0")" && pwd)"                          # .../hardware
NAME=qat6_seed1; FULL=0
for arg in "$@"; do [ "$arg" == "--full" ] && FULL=1 || NAME="$arg"; done
DESIGN="$HW/$NAME"
TOP=transformer

[ -d "$DESIGN/source" ] || { echo "no $DESIGN/source: run convert.py first"; exit 1; }
cd "$DESIGN"                       # the testbench reads ./data/..., so run from here
rm -rf xsim.dir sim_logs .Xil ./*.pb ./*.jou ./*.wdb ./*.log data/${TOP}_q_out.txt
mkdir -p sim_logs

# file list in compile order (innermost entities first), see software/hw_design/vhdl_order.py
(cd "$HW/../software" && uv run python -m hw_design.vhdl_order "$DESIGN/source") > sim_logs/files.txt
echo "1/4 compile    $(wc -l < sim_logs/files.txt) VHDL files"
"$VIVADO_BIN/xvhdl" -2008 -f sim_logs/files.txt > sim_logs/xvhdl.log 2>&1 \
    || { grep ERROR sim_logs/xvhdl.log; exit 1; }

echo "2/4 elaborate  ${TOP}_tb"
"$VIVADO_BIN/xelab" -debug off "${TOP}_tb" -s tb > sim_logs/xelab.log 2>&1 \
    || { grep ERROR sim_logs/xelab.log; exit 1; }

if [ "$FULL" == 1 ]; then
    # one simulation per shard, all at the same time; each shard gets its own copy of the
    # compiled simulation (xsim writes its logs into it) and its own ./data folder
    [ -f full/manifest.json ] || { echo "no full/manifest.json: run convert.py --full first"; exit 1; }
    shards=(full/shard_*)
    total=$(cat full/shard_*/data/${TOP}_q_y.txt | grep -o -- '-\?[0-9]\+' | wc -l)   # files have no final newline
    echo "3/4 simulate   ${#shards[@]} shards in parallel, $total windows (started $(date +%H:%M))"
    printf '%s\n' "${shards[@]}" | xargs -P "$(nproc)" -I{} bash -c '
        cd "$1" && rm -rf xsim.dir data/'"$TOP"'_q_out.txt && cp -r ../../xsim.dir .
        "$2" tb -R > xsim_shard.log 2>&1 || true
        rm -rf xsim.dir ./*.pb ./*.jou xsim.log
        echo "    done $(basename "$1") $(date +%H:%M)"' _ {} "$VIVADO_BIN/xsim"
    rm -f ./*.pb ./*.jou ./*.log
    grep -h "Time taken" full/shard_00/xsim_shard.log | sed 's/^Note: /    /' || true
    echo "4/4 compare + F1 from the hardware outputs"
    (cd "$HW/../software" && uv run python -m hw_design.hw_eval "$DESIGN")
    exit $?
fi

echo "3/4 simulate   (about 2-3 s per test window)"
"$VIVADO_BIN/xsim" tb -R > sim_logs/xsim.log 2>&1 || true   # the testbench always ends with an intended "Failure"
rm -f ./*.pb ./*.jou ./*.log                                  # Vivado's own copies of the logs (ours are in sim_logs/)
grep -E "Correct/Simulated" sim_logs/xsim.log | sed 's/^Note: /    /' || true
grep -E "Time taken" sim_logs/xsim.log | sed 's/^Note: /    /' || true

echo "4/4 compare    VHDL output vs Python integer model"
values() { grep -o -- '-\?[0-9]\+' "$1"; }    # one number per line (Python writes no final newline)
expected=$(values data/${TOP}_q_y.txt)
if [ -f data/${TOP}_q_out.txt ] && [ "$expected" == "$(values data/${TOP}_q_out.txt)" ]; then
    echo "PASS: all $(echo "$expected" | wc -l) outputs identical"
else
    echo "FAIL: outputs differ (or the simulation did not finish), see $DESIGN/sim_logs/xsim.log"
    exit 1
fi
