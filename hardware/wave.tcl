#!/usr/bin/env tclsh
# Waveforms for a few test windows in the Vivado simulator GUI.
#
#   tclsh hardware/wave.tcl        # 2 windows around the fault of valve1/0 (1 normal, 1 fault)
#   tclsh hardware/wave.tcl 4      # 4 windows
#
# Steps: 1 convert.py -> hardware/qat6_seed1_wave (a separate folder: the verified
# design in hardware/qat6_seed1 stays untouched) -> 2 compile order (vhdl_order.py)
# -> 3 compile (xvhdl) -> 4 elaborate WITH debug info (xelab -debug typical, needed to
# see signals) -> 5 open the simulator GUI, record every signal, show the main ones, run.

set windows [expr {$argc > 0 ? [lindex $argv 0] : 2}]
set vivado  [expr {[info exists env(VIVADO_BIN)] ? $env(VIVADO_BIN) : "$env(HOME)/Vivado/2025.2/Vivado/bin"}]
set hw      [file normalize [file dirname [info script]]]
set sw      [file normalize $hw/../software]
set design  $hw/qat6_seed1_wave

# run a command, show its output, stop at the first error
proc run {args} {
    puts "\n>> [join $args]"
    if {[catch {exec {*}$args >@ stdout 2>@ stderr} err]} { puts "FAILED: $err"; exit 1 }
}

puts "1/5 convert   $windows windows -> $design"
cd $sw
run uv run python -m hw_design.convert --windows $windows --out $design

puts "2/5 compile order"
if {[catch {exec uv run python -m hw_design.vhdl_order $design/source > $design/files.txt} err]} {
    puts "FAILED: $err"; exit 1
}

cd $design
puts "3/5 compile (xvhdl)"
run $vivado/xvhdl -2008 -f files.txt
puts "4/5 elaborate with debug info (xelab)"
run $vivado/xelab -debug typical transformer_tb -s tb_wave

# commands the GUI runs at start-up
set f [open wave_gui.tcl w]
puts $f {
log_wave -recursive *            ;# record every signal: any one can be added later from the Scope panel

# testbench: clock, start, done, output
add_wave /transformer_tb/clock /transformer_tb/uut_enable /transformer_tb/done
add_wave -radix dec /transformer_tb/y

# the layer chain: each done starts the next layer
add_wave /transformer_tb/uut/input_linear_done /transformer_tb/uut/add_pos_info_done \
         /transformer_tb/uut/encoder_done /transformer_tb/uut/gap_done /transformer_tb/uut/output_linear_done

# inside input_linear: state machine, input address, accumulator, stored output
add_wave /transformer_tb/uut/inst_input_linear/mac_state /transformer_tb/uut/inst_input_linear/x_address
add_wave -radix dec /transformer_tb/uut/inst_input_linear/macc_sum /transformer_tb/uut/inst_input_linear/y_store_data

run all
}
close $f

puts "5/5 simulator GUI (runs ~15 s, stops at the testbench's intended \"Failure\"; then Zoom Fit)"
puts "    expected y per window: [string map {"\n" " "} [string trim [exec cat data/transformer_q_y.txt]]]"
run $vivado/xsim tb_wave -gui -tclbatch wave_gui.tcl
