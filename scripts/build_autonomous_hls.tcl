set script_dir [file dirname [file normalize [info script]]]
set project_dir [file normalize [file join $script_dir ..]]
set build_dir [file join $project_dir build autonomous_hls]
set source_dir [file join $project_dir rtl autonomous]
set hls_project tinyllm_autonomous_i8
if {[info exists ::env(AUTONOMOUS_HLS_PROJECT)]} {
    set hls_project $::env(AUTONOMOUS_HLS_PROJECT)
}
set package_dir [file normalize [file join $project_dir build pure_ps fused_ps128_candidate]]
if {[info exists ::env(AUTONOMOUS_PACKAGE_DIR)]} {
    set package_dir [file normalize $::env(AUTONOMOUS_PACKAGE_DIR)]
}
set python python
if {[info exists ::env(AUTONOMOUS_PYTHON)]} {
    set python $::env(AUTONOMOUS_PYTHON)
}
set layout_cmd [list $python -B [file join $project_dir scripts generate_hls_layout.py] \
    --manifest [file join $package_dir manifest.json] \
    --weights [file join $package_dir model_q6w.bin] \
    --model-output [file join $package_dir autonomous_model.bin] \
    --output [file join $source_dir model_layout.h]]
puts "Preparing aligned int8 model image for autonomous PL inference"
exec {*}$layout_cmd

file mkdir $build_dir
cd $build_dir
open_project -reset $hls_project
set_top tinyllm_autonomous
set source_flags "-std=c++0x -I$source_dir"
if {[info exists ::env(AUTONOMOUS_PERFORMANCE_BUILD)] && $::env(AUTONOMOUS_PERFORMANCE_BUILD) eq "1"} {
    append source_flags " -DTINYLLM_PERFORMANCE_BUILD=1"
}
if {[info exists ::env(AUTONOMOUS_LEGACY_KV_MIRROR)] && $::env(AUTONOMOUS_LEGACY_KV_MIRROR) eq "1"} {
    append source_flags " -DTINYLLM_LEGACY_KV_MIRROR=1"
}
if {[info exists ::env(AUTONOMOUS_PARALLEL_FFN)] && $::env(AUTONOMOUS_PARALLEL_FFN) eq "1"} {
    if {![info exists ::env(AUTONOMOUS_LEGACY_KV_MIRROR)] || $::env(AUTONOMOUS_LEGACY_KV_MIRROR) ne "1"} {
        error "AUTONOMOUS_PARALLEL_FFN requires AUTONOMOUS_LEGACY_KV_MIRROR=1"
    }
    append source_flags " -DTINYLLM_PARALLEL_FFN=1"
}
set precomputed_rope 0
if {[info exists ::env(AUTONOMOUS_PRECOMPUTED_ROPE)] && $::env(AUTONOMOUS_PRECOMPUTED_ROPE) eq "1"} {
    set precomputed_rope 1
    append source_flags " -DTINYLLM_PRECOMPUTED_ROPE=1"
}
if {[info exists ::env(AUTONOMOUS_PARALLEL_KV)] && $::env(AUTONOMOUS_PARALLEL_KV) eq "1"} {
    if {![info exists ::env(AUTONOMOUS_LEGACY_KV_MIRROR)] || $::env(AUTONOMOUS_LEGACY_KV_MIRROR) ne "1"} {
        error "AUTONOMOUS_PARALLEL_KV requires AUTONOMOUS_LEGACY_KV_MIRROR=1"
    }
    if {![info exists ::env(AUTONOMOUS_PARALLEL_FFN)] || $::env(AUTONOMOUS_PARALLEL_FFN) ne "1"} {
        error "AUTONOMOUS_PARALLEL_KV requires AUTONOMOUS_PARALLEL_FFN=1"
    }
    if {!$precomputed_rope} {
        error "AUTONOMOUS_PARALLEL_KV requires AUTONOMOUS_PRECOMPUTED_ROPE=1 to free DSP resources"
    }
    append source_flags " -DTINYLLM_PARALLEL_KV=1"
}
add_files [file join $source_dir tinyllm_autonomous.cpp] -cflags $source_flags
add_files -tb [file join $source_dir tb_tinyllm_autonomous.cpp] -cflags $source_flags
open_solution -reset solution_z7020
set_part {xc7z020clg400-1}
create_clock -period 8 -name default
set csim_args [list \
    [file join $package_dir autonomous_model.bin] \
    [file join $package_dir hls_prompt.bin] \
    [file join $package_dir reference_logits.bin]]
if {$precomputed_rope} {
    lappend csim_args [file join $package_dir rope_table.bin]
}
csim_design -argv $csim_args
if {[info exists ::env(AUTONOMOUS_CSIM_ONLY)] && $::env(AUTONOMOUS_CSIM_ONLY) eq "1"} {
    puts "CSIM_ONLY_COMPLETE"
    exit
}
csynth_design
if {[info exists ::env(AUTONOMOUS_CSYNTH_ONLY)] && $::env(AUTONOMOUS_CSYNTH_ONLY) eq "1"} {
    puts "CSYNTH_ONLY_COMPLETE"
    exit
}
export_design -format ip_catalog -vendor llm-pynq.org -library user -version 1.0
exit
