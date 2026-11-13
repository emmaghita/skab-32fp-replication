# Synthesis + implementation of a generated design for the Z-turn's FPGA (xc7z020clg400-1), in
# Vivado batch mode: what the transformer costs on the chip and whether it meets 100 MHz.
#
#   ~/Vivado/2025.2/Vivado/bin/vivado -mode batch -nojournal -log hardware/synth_qat6_seed1.log \
#       -source hardware/synth.tcl -tclargs qat6_seed1
#
# Out-of-context: the transformer alone, NOT connected to the chip's pins. On the board it sits
# inside an AXI wrapper (zynq-shell), so this is the cost of the transformer itself.
# No bitstream: that needs the whole system (PS7 + wrapper), built later by `zb build`.
#
# Steps: 1 read the VHDL (no testbenches) -> 2 clock constraint (100 MHz, like zynq-shell's FCLK0)
# -> 3 synth_design -> 4 opt / place / route -> 5 reports -> 6 netlist for post-synthesis simulation.
# Output: hardware/<design>/synth/  (reports/*.rpt with the same names as zynq-shell's build.tcl,
# summary.txt, post_route.dcp, transformer_funcsim.vhd)

set name     [expr {$argc > 0 ? [lindex $argv 0] : "qat6_seed1"}]
set part     xc7z020clg400-1
set top      transformer
set period   10.0                                    ;# ns -> 100 MHz
set hw       [file normalize [file dirname [info script]]]
set design   $hw/$name
set out      $design/synth
if {![file isdirectory $design/source]} { error "no $design/source: run convert.py first" }
file delete -force $out
file mkdir $out/reports
set_param general.maxThreads 8

# 1) design files in compile order (vhdl_order.py), without the 24 testbenches (simulation only)
set here [pwd]
cd $hw/../software
# (without Vivado's own PYTHONHOME/PYTHONPATH/LD_LIBRARY_PATH, which break the project's Python)
set files [exec env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH \
               uv run python -m hw_design.vhdl_order $design/source]
cd $here
set n 0
foreach f [split $files "\n"] {
    if {$f eq "" || [string match "*_tb.vhd" $f]} { continue }
    read_vhdl -vhdl2008 $f
    incr n
}
puts "synth: $n design files"

# 2) the clock: every path must finish within one 10 ns period
set fh [open $out/clock.xdc w]
puts $fh "create_clock -name clk -period $period \[get_ports clock\]"
close $fh
read_xdc -mode out_of_context $out/clock.xdc

# 3) synthesis: VHDL -> LUTs, flip-flops, DSPs, block RAM
synth_design -top $top -part $part -mode out_of_context
report_utilization -file $out/reports/utilization_synth.rpt
write_vhdl -force -mode funcsim $out/${top}_funcsim.vhd        ;# for the post-synthesis simulation

# 4) implementation: place everything on the chip and route the real wires (real delays)
opt_design
place_design
route_design
write_checkpoint -force $out/post_route.dcp

# 5) reports (after routing: the timing is real now)
report_utilization                -file $out/reports/utilization.rpt
report_utilization -hierarchical  -file $out/reports/utilization_hier.rpt
report_timing_summary -max_paths 10 -file $out/reports/timing.rpt
report_power                      -file $out/reports/power.rpt
report_methodology                -file $out/reports/methodology.rpt

# summary: resources, timing, power, latches
proc util {text label} {   ;# "| Slice LUTs | used | fixed | prohibited | available | util% |"
    if {[regexp "\\|\\s*${label}\\s*\\|\\s*(\[0-9.\]+)\\s*\\|\[^|\]*\\|\[^|\]*\\|\\s*(\[0-9.\]+)\\s*\\|\\s*(\[0-9.<\]+)\\s*\\|" $text -> used avail pct]} {
        return [format "%-22s %8s / %-7s (%s %%)" $label $used $avail $pct]
    }
    return "$label: not found"
}
set u   [report_utilization -return_string]
set pw  [report_power -return_string]
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]
set fmax [expr {1000.0 / ($period - $wns)}]
regexp {Total On-Chip Power \(W\)\s*\|\s*([0-9.]+)} $pw -> p_total
regexp {Dynamic \(W\)\s*\|\s*([0-9.]+)} $pw -> p_dyn
regexp {Device Static \(W\)\s*\|\s*([0-9.]+)} $pw -> p_static
set latches [llength [get_cells -hierarchical -quiet -filter {PRIMITIVE_TYPE =~ REGISTER.LATCH.*}]]

set lines [list \
    "design   : $name  (top $top, part $part, out-of-context, clock [expr {1000.0 / $period}] MHz)" \
    [util $u "Slice LUTs"] [util $u "LUT as Logic"] [util $u "LUT as Memory"] \
    [util $u "Slice Registers"] [util $u "Block RAM Tile"] [util $u "DSPs"] \
    [format "timing   : WNS %.3f ns, WHS %.3f ns -> %s, Fmax ~%.1f MHz" $wns $whs \
        [expr {$wns >= 0 && $whs >= 0 ? "100 MHz MET" : "100 MHz NOT MET"}] $fmax] \
    [format "latency  : ~227,000 cycles (simulation) -> 2.27 ms at 100 MHz, %.2f ms at Fmax" [expr {227000 / ($fmax * 1000.0)}]] \
    "power    : total $p_total W = static $p_static W + dynamic $p_dyn W (vectorless estimate)" \
    "latches  : $latches" ]
set fh [open $out/reports/summary.txt w]
foreach l $lines { puts $fh $l }
close $fh
puts "\n================ synth.tcl summary ================"
foreach l $lines { puts $l }
puts "reports  : $out/reports/"
