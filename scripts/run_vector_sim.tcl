set script_dir [file dirname [file normalize [info script]]]
set demo_dir [file normalize [file join $script_dir ..]]
if {[llength $argv] != 1} {
    error "Usage: run_vector_sim.tcl <vector-directory>"
}
set vector_dir [file normalize [lindex $argv 0]]
set sim_dir [file join $vector_dir sim]
file mkdir $sim_dir
foreach required {input.hex expected.hex} {
    if {![file exists [file join $vector_dir $required]]} {
        error "Missing vector file: [file join $vector_dir $required]"
    }
}

set vivado_root [file normalize [file join [file dirname [info nameofexecutable]] .. .. ..]]
set xvlog [file join $vivado_root bin xvlog.bat]
set xelab [file join $vivado_root bin xelab.bat]
set xsim [file join $vivado_root bin xsim.bat]
set rtl_file [file join $demo_dir rtl q6w_gemv_axis.sv]
set tb_file [file join $demo_dir sim tb_q6w_gemv.sv]

file delete -force [file join $sim_dir xsim.dir]
cd $sim_dir
set compile_cmd [list $xvlog -sv $rtl_file $tb_file]
if {[catch {exec {*}$compile_cmd 2>@1} output]} {
    puts stderr $output
    error "xvlog failed"
}
puts $output

set elaborate_cmd [list $xelab work.tb_q6w_gemv -s q6w_vector_sim]
if {[catch {exec {*}$elaborate_cmd 2>@1} output]} {
    puts stderr $output
    error "xelab failed"
}
puts $output

file copy -force [file join $vector_dir input.hex] [file join $sim_dir input.hex]
file copy -force [file join $vector_dir expected.hex] [file join $sim_dir expected.hex]
set simulate_cmd [list $xsim q6w_vector_sim -runall]
if {[catch {exec {*}$simulate_cmd 2>@1} output]} {
    puts stderr $output
    error "xsim failed"
}
puts $output
if {[string first "PASS:" $output] < 0} {
    error "Simulation did not report PASS"
}
