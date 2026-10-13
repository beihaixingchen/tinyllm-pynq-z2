set script_dir [file dirname [file normalize [info script]]]
set demo_dir [file normalize [file join $script_dir ..]]
set sim_dir [file join $demo_dir build sim]
file mkdir $sim_dir

set weights [file normalize [file join $demo_dir .. .. TinyLLMStory-zh quantization artifacts tinyllamas_zh_story_q6w_qat_lr1e6.bin]]
set manifest [file normalize [file join $demo_dir .. .. TinyLLMStory-zh quantization artifacts tinyllamas_zh_story_q6w_qat_lr1e6.json]]
set generator [file join $script_dir prepare_vectors.py]
set python [auto_execok python]
if {$python eq ""} {
    error "Python 3 was not found on PATH"
}

set prepare_cmd [list {*}$python $generator --weights $weights --manifest $manifest --out-dir $sim_dir]
if {[catch {exec {*}$prepare_cmd 2>@1} output]} {
    puts stderr $output
    error "Failed to generate Q6W test vectors"
}
puts $output

set rtl_file [file join $demo_dir rtl q6w_gemv_axis.sv]
set tb_file [file join $script_dir .. sim tb_q6w_gemv.sv]
set vivado_root [file normalize [file join [file dirname [info nameofexecutable]] .. .. ..]]
set xvlog [file join $vivado_root bin xvlog.bat]
set xelab [file join $vivado_root bin xelab.bat]
set xsim [file join $vivado_root bin xsim.bat]

file delete -force [file join $sim_dir xsim.dir]
cd $sim_dir

set compile_cmd [list $xvlog -sv $rtl_file $tb_file]
if {[catch {exec {*}$compile_cmd 2>@1} output]} {
    puts stderr $output
    error "xvlog failed"
}
puts $output

set elaborate_cmd [list $xelab work.tb_q6w_gemv -s q6w_gemv_sim]
if {[catch {exec {*}$elaborate_cmd 2>@1} output]} {
    puts stderr $output
    error "xelab failed"
}
puts $output

set simulate_cmd [list $xsim q6w_gemv_sim -runall]
if {[catch {exec {*}$simulate_cmd 2>@1} output]} {
    puts stderr $output
    error "xsim failed"
}
puts $output
if {[string first "PASS:" $output] < 0} {
    error "Simulation did not report PASS"
}
