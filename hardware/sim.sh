#!/usr/bin/env bash
# Simulate a generated design in Vivado's simulator and check it against the Python integer model.
#
#   ./hardware/sim.sh                     # default design folder: hardware/qat6_seed1
#   ./hardware/sim.sh qat6_seed0          # another folder made by convert.py
#   ./hardware/sim.sh qat6_seed1 --full   # ALL validation + test windows (convert.py --full),
#                                         # in parallel, then F1 from the hardware outputs
#   ./hardware/sim.sh qat6_seed1 --post 1 # POST-SYNTHESIS simulation of the first N windows of
#                                         # data/ (needs synth.tcl first), one simulation per window
#
# Steps: 1 compile (xvhdl) -> 2 elaborate (xelab) -> 3 run the testbench (xsim) -> 4 compare.
# PASS = the VHDL outputs (data/transformer_q_out.txt) equal the Python integer outputs
# (data/transformer_q_y.txt) for every test window.
set -euo pipefail

VIVADO_BIN="${VIVADO_BIN:-$HOME/Vivado/2025.2/Vivado/bin}"   # override: VIVADO_BIN=... ./sim.sh
HW="$(cd "$(dirname "$0")" && pwd)"                          # .../hardware
NAME=qat6_seed1; FULL=0; POST=0
while [ $# -gt 0 ]; do
    case "$1" in
        --full) FULL=1 ;;
        --post) if [[ "${2:-}" =~ ^[0-9]+$ ]]; then POST=$2; shift; else POST=1; fi ;;
        *)      NAME="$1" ;;
    esac
    shift
done
DESIGN="$HW/$NAME"
TOP=transformer
values() { grep -o -- '-\?[0-9]\+' "$1"; }    # one number per line (Python writes no final newline)

[ -d "$DESIGN/source" ] || { echo "no $DESIGN/source: run convert.py first"; exit 1; }

if [ "$POST" -gt 0 ]; then
    # Post-synthesis (functional) simulation: the same testbench, but around the netlist that
    # synth.tcl wrote (LUTs, flip-flops, DSPs, BRAMs from Xilinx's unisim library) instead of
    # our VHDL. Checks that synthesis did not change what the design computes. No wire delays.
    # Gate level is slow and xsim uses one core, so: one simulation per window, all in parallel.
    NETLIST="$DESIGN/synth/${TOP}_funcsim.vhd"
    [ -f "$NETLIST" ] || { echo "no $NETLIST: run synth.tcl first"; exit 1; }
    available=$(values "$DESIGN/data/${TOP}_q_y.txt" | wc -l)
    [ "$POST" -le "$available" ] || { echo "--post $POST: data/ has only $available windows"; exit 1; }
    POSTDIR="$DESIGN/postsim"
    rm -rf "$POSTDIR"; mkdir -p "$POSTDIR"; cd "$POSTDIR"

    # the testbench instantiates transformer(rtl); the netlist's architecture is "STRUCTURE"
    sed 's/entity work.transformer(rtl)/entity work.transformer(STRUCTURE)/' \
        "$DESIGN/source/$TOP/${TOP}_tb.vhd" > ${TOP}_tb.vhd
    grep -q "(STRUCTURE)" ${TOP}_tb.vhd || { echo "testbench: uut line not found"; exit 1; }
    # the clock must start LOW: ElasticAI's clock generator starts with clock <= '1', i.e. a
    # rising edge at t = 0, before any gate has an output. Two softmax flip-flops without reset
    # (calculate_denominator's state/pre_fetch) store that X and keep it -> the softmax hangs.
    # Simulation only: on the chip flip-flops start at 0. (RTL is unaffected: there they start
    # at their declared values.)
    perl -0pi -e "s/clock <= '1';(\s*wait for C_CLK_PERIOD\/2;\s*)clock <= '0';/clock <= '0';\1clock <= '1';/" ${TOP}_tb.vhd
    grep -qPz "clock <= '0';\s*wait for C_CLK_PERIOD/2;\s*clock <= '1';" ${TOP}_tb.vhd \
        || { echo "testbench: clock generator not found"; exit 1; }
    X_COUNT=$(grep -oP 'X_COUNT : integer := \K[0-9]+' ${TOP}_tb.vhd)   # input values per window

    echo "1/4 compile    synthesized netlist + testbench"
    { "$VIVADO_BIN/xvhdl" "$NETLIST" && "$VIVADO_BIN/xvhdl" -2008 ${TOP}_tb.vhd; } > xvhdl.log 2>&1 \
        || { grep ERROR xvhdl.log; exit 1; }
    echo "2/4 elaborate  ${TOP}_tb with unisim (Xilinx's models of the chip's primitives)"
    "$VIVADO_BIN/xelab" -debug off -L unisim ${TOP}_tb -s tb_post > xelab.log 2>&1 \
        || { grep ERROR xelab.log; exit 1; }

    # window i -> shard_i/data with its X_COUNT input lines and its 1 expected output
    for ((i = 0; i < POST; i++)); do
        d=shard_$(printf %02d $i); mkdir -p $d/data
        values "$DESIGN/data/${TOP}_q_x.txt" | sed -n "$((i * X_COUNT + 1)),$(((i + 1) * X_COUNT))p" \
            > $d/data/${TOP}_q_x.txt
        values "$DESIGN/data/${TOP}_q_y.txt" | sed -n "$((i + 1))p" > $d/data/${TOP}_q_y.txt
    done

    JOBS="${JOBS:-$(nproc)}"           # override: JOBS=8 ./sim.sh ... --post 16
    echo "3/4 simulate   $POST windows, up to $JOBS at a time (started $(date +%H:%M))"
    start=$(date +%s)
    printf '%s\n' shard_* | xargs -P "$JOBS" -I{} bash -c '
        cd "$1" && cp -r ../xsim.dir . && t0=$(date +%s)
        "$2" tb_post -R > xsim_shard.log 2>&1 || true
        rm -rf xsim.dir ./*.pb ./*.jou xsim.log
        echo "    done $1 in $(( $(date +%s) - t0 )) s"' _ {} "$VIVADO_BIN/xsim" &
    runner=$!
    peak=0                             # largest memory of one simulation, sampled every 2 s
    while kill -0 $runner 2>/dev/null; do
        r=$(ps -C xsimk -o rss= 2>/dev/null | sort -n | tail -1 | tr -d ' ') || true
        [ -n "$r" ] && [ "$r" -gt "$peak" ] && peak=$r
        sleep 2
    done
    wait $runner || true
    rm -f ./*.pb ./*.jou
    echo "    total $(( $(date +%s) - start )) s, peak RAM per simulation ~$((peak / 1024)) MB"

    echo "4/4 compare    netlist output vs Python integer model"
    got=$(for d in shard_*; do values $d/data/${TOP}_q_out.txt 2>/dev/null || echo missing; done)
    expected=$(values "$DESIGN/data/${TOP}_q_y.txt" | head -n "$POST")
    if [ "$got" == "$expected" ]; then
        echo "PASS: all $POST outputs identical (post-synthesis)"
    else
        echo "FAIL: outputs differ (or a simulation did not finish), see $POSTDIR/shard_*/xsim_shard.log"
        paste <(echo "$expected") <(echo "$got") | awk '{print "    window " NR-1 ": python " $1 ", netlist " $2}'
        exit 1
    fi
    exit 0
fi

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
expected=$(values data/${TOP}_q_y.txt)
if [ -f data/${TOP}_q_out.txt ] && [ "$expected" == "$(values data/${TOP}_q_out.txt)" ]; then
    echo "PASS: all $(echo "$expected" | wc -l) outputs identical"
else
    echo "FAIL: outputs differ (or the simulation did not finish), see $DESIGN/sim_logs/xsim.log"
    exit 1
fi
