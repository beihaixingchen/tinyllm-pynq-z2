set script_dir [file dirname [file normalize [info script]]]
set demo_dir [file normalize [file join $script_dir ..]]
if {[llength $argv] < 1 || [llength $argv] > 2} {
    error "Usage: run_batched_sim.tcl <vector-directory> ?base|pipe|pipe32?"
}
set vector_dir [file normalize [lindex $argv 0]]
set rtl_variant base
if {[llength $argv] == 2} {
    set rtl_variant [lindex $argv 1]
}
if {$rtl_variant ni {base pipe pipe32}} {
    error "RTL variant must be base, pipe, or pipe32"
}
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
set glbl_file [file join $vivado_root data verilog src glbl.v]
set tb_file [file join $demo_dir sim tb_q6w_gemv_batched.sv]
if {$rtl_variant in {pipe pipe32}} {
    set rtl_file [file join $demo_dir rtl q6w_gemv_batched_pipe_axis.sv]
    if {$rtl_variant eq "pipe32"} {
        set adapter_file [file join $demo_dir sim q6w_pipe32_cache_adapter.sv]
    } else {
        set adapter_file [file join $demo_dir sim q6w_pipe_cache_adapter.sv]
    }
} else {
    set rtl_file [file join $demo_dir rtl q6w_gemv_batched_axis.sv]
}

file delete -force [file join $sim_dir xsim.dir]
cd $sim_dir
set compile_cmd [list $xvlog -sv $rtl_file]
if {$rtl_variant in {pipe pipe32}} {
    lappend compile_cmd $adapter_file
}
if {$rtl_variant eq "pipe32"} {
    lappend compile_cmd $glbl_file
}
lappend compile_cmd $tb_file
if {[catch {exec {*}$compile_cmd 2>@1} output]} {
    puts stderr $output
    error "xvlog failed"
}
puts $output

set elaborate_cmd [list $xelab]
if {$rtl_variant eq "pipe32"} {
    lappend elaborate_cmd -L unisims_ver
}
lappend elaborate_cmd work.tb_q6w_gemv_batched
if {$rtl_variant eq "pipe32"} {
    lappend elaborate_cmd work.glbl
}
lappend elaborate_cmd -s q6_batched_sim
if {[catch {exec {*}$elaborate_cmd 2>@1} output]} {
    puts stderr $output
    error "xelab failed"
}
puts $output

file copy -force [file join $vector_dir input.hex] [file join $sim_dir input.hex]
file copy -force [file join $vector_dir expected.hex] [file join $sim_dir expected.hex]
set simulate_cmd [list $xsim q6_batched_sim -runall]
if {[catch {exec {*}$simulate_cmd 2>@1} output]} {
    puts stderr $output
    error "xsim failed"
}
puts $output
if {[string first "PASS:" $output] < 0} {
    error "Simulation did not report PASS"
}
