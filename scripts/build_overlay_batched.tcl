set script_dir [file dirname [file normalize [info script]]]
set demo_dir [file normalize [file join $script_dir ..]]
if {[llength $argv] > 6} {
    error "Usage: build_overlay_batched.tcl ?system-clock-MHz? ?build-tag? ?cache-profile? ?base|pipe? ?16|32-cache-lanes? ?core-clock-MHz?"
}
set target_mhz 100
set build_tag batched_linears
set cache_profile ffn
set implementation base
set cache_lanes 16
set core_target_mhz 0
if {[llength $argv] == 1} {
    set target_mhz [lindex $argv 0]
} elseif {[llength $argv] == 2} {
    set target_mhz [lindex $argv 0]
    set build_tag [lindex $argv 1]
} elseif {[llength $argv] == 3} {
    set target_mhz [lindex $argv 0]
    set build_tag [lindex $argv 1]
    set cache_profile [lindex $argv 2]
} elseif {[llength $argv] == 4} {
    set target_mhz [lindex $argv 0]
    set build_tag [lindex $argv 1]
    set cache_profile [lindex $argv 2]
    set implementation [lindex $argv 3]
} elseif {[llength $argv] == 5} {
    set target_mhz [lindex $argv 0]
    set build_tag [lindex $argv 1]
    set cache_profile [lindex $argv 2]
    set implementation [lindex $argv 3]
    set cache_lanes [lindex $argv 4]
} elseif {[llength $argv] == 6} {
    set target_mhz [lindex $argv 0]
    set build_tag [lindex $argv 1]
    set cache_profile [lindex $argv 2]
    set implementation [lindex $argv 3]
    set cache_lanes [lindex $argv 4]
    set core_target_mhz [lindex $argv 5]
}
if {![string is integer -strict $target_mhz] || $target_mhz < 80 || $target_mhz > 250} {
    error "PL clock must be an integer from 80 through 250 MHz"
}
if {![regexp {^[A-Za-z0-9_]+$} $build_tag]} {
    error "Build tag must contain only letters, digits, and underscores"
}
if {$cache_profile ni {ffn ffn_qkv vocab}} {
    error "Cache profile must be ffn, ffn_qkv, or vocab"
}
if {$implementation ni {base pipe}} {
    error "RTL implementation must be base or pipe"
}
if {$cache_lanes ni {16 32}} {
    error "Cached GEMV parallelism must be 16 or 32 lanes"
}
if {$implementation eq "base" && $cache_lanes != 16} {
    error "The base implementation only supports 16 cache lanes"
}
if {$core_target_mhz != 0 && (![string is integer -strict $core_target_mhz] || $core_target_mhz < 80 || $core_target_mhz > 250)} {
    error "Core clock must be 0 (same as system clock) or an integer from 80 through 250 MHz"
}
if {$core_target_mhz != 0 && $implementation ne "pipe"} {
    error "A separate core clock is supported only by the pipelined implementation"
}
set core_module q6w_gemv_batched_axis
set core_display {Q6W Batched AXI Stream GEMV}
if {$implementation eq "pipe"} {
    set core_module q6w_gemv_batched_pipe_axis
    set core_display {Q6W Pipelined Batched AXI Stream GEMV}
}
set rtl_file [file join $demo_dir rtl ${core_module}.sv]
set cache_slots 10
set cache_words 112640
set cache_matrix_rows 352
if {$cache_profile eq "vocab"} {
    set cache_slots 1
    set cache_words 131072
    set cache_matrix_rows 4096
} elseif {$cache_profile eq "ffn_qkv"} {
    set cache_slots 15
    set cache_words 168960
    set cache_matrix_rows 352
}
set build_dir [file join $demo_dir build ${build_tag}_${target_mhz}MHz]
set project_dir [file join $build_dir vivado]
set ip_project_dir [file join $build_dir ip_packager]
set ip_repo_dir [file join $build_dir ip_repo]
set project_name q6w_gemv_batched
if {$build_tag ne "batched_linears"} {
    set project_name q6w_gemv_${build_tag}
}

file delete -force $ip_project_dir $ip_repo_dir
create_project ${project_name}_ip $ip_project_dir -part xc7z020clg400-1 -force
set_property source_mgmt_mode None [current_project]
add_files -norecurse $rtl_file
set rtl_obj [get_files $rtl_file]
set_property file_type Verilog $rtl_obj
set_property top $core_module [get_filesets sources_1]
update_compile_order -fileset sources_1

ipx::package_project -root_dir $ip_repo_dir -vendor llm-pynq.org \
    -library user -taxonomy /UserIP -import_files -set_current true
set core [ipx::current_core]
set_property name $core_module $core
set_property version 1.0 $core
set_property display_name $core_display $core
set_property description {Batched runtime-sized signed int6 by int16 GEMV, K up to 352 and rows up to 4096} $core
ipx::infer_bus_interface {s_axis_tdata s_axis_tvalid s_axis_tready s_axis_tlast} \
    xilinx.com:interface:axis_rtl:1.0 $core
ipx::infer_bus_interface {m_axis_tdata m_axis_tvalid m_axis_tready m_axis_tlast} \
    xilinx.com:interface:axis_rtl:1.0 $core
ipx::infer_bus_interface {clk} xilinx.com:signal:clock_rtl:1.0 $core
ipx::infer_bus_interface {rst_n} xilinx.com:signal:reset_rtl:1.0 $core
ipx::save_core $core
close_project

file delete -force $project_dir
create_project $project_name $project_dir -part xc7z020clg400-1 -force
set_property board_part tul.com.tw:pynq-z2:part0:1.0 [current_project]
set_property source_mgmt_mode None [current_project]
set_property ip_repo_paths [list $ip_repo_dir] [current_project]
update_ip_catalog

create_bd_design system
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} \
    [get_bd_cells processing_system7_0]
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP1 {1} \
    CONFIG.PCW_EN_CLK0_PORT {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ $target_mhz \
    CONFIG.PCW_EN_CLK1_PORT [expr {$core_target_mhz != 0}] \
    CONFIG.PCW_FPGA1_PERIPHERAL_FREQMHZ [expr {$core_target_mhz != 0 ? $core_target_mhz : $target_mhz}] \
    CONFIG.PCW_EN_RST0_PORT {1}] [get_bd_cells processing_system7_0]

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
# S2MM destinations are PYNQ CMA buffers and output words are 32-bit aligned.
# Keep DRE disabled; transfers are kept below the 16-bit BTT limit.
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_include_mm2s {1} \
    CONFIG.c_include_s2mm {1} \
    CONFIG.c_include_s2mm_dre {0} \
    CONFIG.c_sg_length_width {16} \
    CONFIG.c_m_axis_mm2s_tdata_width {8} \
    CONFIG.c_mm2s_burst_size {16} \
    CONFIG.c_s2mm_burst_size {16}] [get_bd_cells axi_dma_0]

create_bd_cell -type ip -vlnv llm-pynq.org:user:${core_module}:1.0 q6w_gemv_0
set core_properties [list \
    CONFIG.CACHE_SLOTS $cache_slots \
    CONFIG.CACHE_WORDS $cache_words \
    CONFIG.CACHE_MATRIX_ROWS $cache_matrix_rows]
if {$implementation eq "pipe"} {
    lappend core_properties CONFIG.CACHE_LANES $cache_lanes
}
set_property -dict $core_properties [get_bd_cells q6w_gemv_0]
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_100M
if {$core_target_mhz != 0} {
    create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_core
    create_bd_cell -type ip -vlnv xilinx.com:ip:axis_clock_converter:1.1 axis_mm2s_clock_converter
    create_bd_cell -type ip -vlnv xilinx.com:ip:axis_clock_converter:1.1 axis_s2mm_clock_converter
    set_property -dict [list \
        CONFIG.IS_ACLK_ASYNC {1} \
        CONFIG.TDATA_NUM_BYTES {1}] [get_bd_cells axis_mm2s_clock_converter]
    set_property -dict [list \
        CONFIG.IS_ACLK_ASYNC {1} \
        CONFIG.TDATA_NUM_BYTES {4}] [get_bd_cells axis_s2mm_clock_converter]
}
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/processing_system7_0/M_AXI_GP0" Clk_master "Auto" Clk_slave "Auto"} \
    [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/axi_dma_0/M_AXI_MM2S" Clk_master "Auto" Clk_slave "Auto"} \
    [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/axi_dma_0/M_AXI_S2MM" Clk_master "Auto" Clk_slave "Auto"} \
    [get_bd_intf_pins processing_system7_0/S_AXI_HP1]

if {$core_target_mhz == 0} {
    connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
        [get_bd_pins q6w_gemv_0/clk]
    connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] \
        [get_bd_pins q6w_gemv_0/rst_n]
    connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] \
        [get_bd_intf_pins q6w_gemv_0/S_AXIS]
    connect_bd_intf_net [get_bd_intf_pins q6w_gemv_0/M_AXIS] \
        [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
} else {
    connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK1] \
        [get_bd_pins rst_ps7_core/slowest_sync_clk] \
        [get_bd_pins q6w_gemv_0/clk] \
        [get_bd_pins axis_mm2s_clock_converter/m_axis_aclk] \
        [get_bd_pins axis_s2mm_clock_converter/s_axis_aclk]
    connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
        [get_bd_pins axis_mm2s_clock_converter/s_axis_aclk] \
        [get_bd_pins axis_s2mm_clock_converter/m_axis_aclk]
    connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] \
        [get_bd_pins rst_ps7_core/ext_reset_in]
    connect_bd_net [get_bd_pins rst_ps7_core/peripheral_aresetn] \
        [get_bd_pins q6w_gemv_0/rst_n] \
        [get_bd_pins axis_mm2s_clock_converter/m_axis_aresetn] \
        [get_bd_pins axis_s2mm_clock_converter/s_axis_aresetn]
    connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] \
        [get_bd_pins axi_dma_0/axi_resetn] \
        [get_bd_pins axis_mm2s_clock_converter/s_axis_aresetn] \
        [get_bd_pins axis_s2mm_clock_converter/m_axis_aresetn]
    connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] \
        [get_bd_intf_pins axis_mm2s_clock_converter/S_AXIS]
    connect_bd_intf_net [get_bd_intf_pins axis_mm2s_clock_converter/M_AXIS] \
        [get_bd_intf_pins q6w_gemv_0/S_AXIS]
    connect_bd_intf_net [get_bd_intf_pins q6w_gemv_0/M_AXIS] \
        [get_bd_intf_pins axis_s2mm_clock_converter/S_AXIS]
    connect_bd_intf_net [get_bd_intf_pins axis_s2mm_clock_converter/M_AXIS] \
        [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
}

assign_bd_address
validate_bd_design
save_bd_design

set actual_mhz [get_property CONFIG.PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ \
    [get_bd_cells processing_system7_0]]
if {![string is double -strict $actual_mhz] || $actual_mhz <= 0} {
    error "Could not determine the actual PS7 FCLK0 frequency: $actual_mhz"
}
set bd_file [get_files [file join $project_dir ${project_name}.srcs sources_1 bd system system.bd]]
generate_target all $bd_file
set ps_xdc [file join $project_dir ${project_name}.srcs sources_1 bd system ip \
    system_processing_system7_0_0 system_processing_system7_0_0.xdc]
if {![file exists $ps_xdc]} {
    error "Missing generated PS7 constraints: $ps_xdc"
}
set fp [open $ps_xdc r]
set xdc_text [read $fp]
close $fp
set clock_period [expr {1000.0 / $actual_mhz}]
set clock_pattern {create_clock -name clk_fpga_0 -period "[0-9]+(\.[0-9]+)?"}
set new_clock [format {create_clock -name clk_fpga_0 -period "%.6f"} $clock_period]
if {![regexp $clock_pattern $xdc_text old_clock]} {
    error "Could not find a generated clk_fpga_0 PS7 clock constraint in $ps_xdc"
}
set xdc_text [string map [list $old_clock $new_clock] $xdc_text]
set fp [open $ps_xdc w]
puts -nonewline $fp $xdc_text
close $fp
puts "PS7_ACTUAL_CLOCK_MHZ=$actual_mhz"
puts "PS7_CLOCK_CONSTRAINT_NS=$clock_period"
if {$core_target_mhz != 0} {
    set actual_core_mhz [get_property CONFIG.PCW_ACT_FPGA1_PERIPHERAL_FREQMHZ \
        [get_bd_cells processing_system7_0]]
    if {![string is double -strict $actual_core_mhz] || $actual_core_mhz <= 0} {
        error "Could not determine the actual PS7 FCLK1 frequency: $actual_core_mhz"
    }
    set core_clock_period [expr {1000.0 / $actual_core_mhz}]
    set core_clock_pattern {create_clock -name clk_fpga_1 -period "[0-9]+(\.[0-9]+)?"}
    set core_new_clock [format {create_clock -name clk_fpga_1 -period "%.6f"} $core_clock_period]
    if {![regexp $core_clock_pattern $xdc_text old_core_clock]} {
        error "Could not find a generated clk_fpga_1 PS7 clock constraint in $ps_xdc"
    }
    set xdc_text [string map [list $old_core_clock $core_new_clock] $xdc_text]
    set fp [open $ps_xdc w]
    puts -nonewline $fp $xdc_text
    close $fp
    puts "PS7_ACTUAL_CORE_CLOCK_MHZ=$actual_core_mhz"
    puts "PS7_CORE_CLOCK_CONSTRAINT_NS=$core_clock_period"
}
set source_set [get_filesets sources_1]
set_property top_auto_set false $source_set
set_property top system $source_set
update_compile_order -fileset sources_1

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
set run_status [get_property STATUS [get_runs impl_1]]
puts "IMPLEMENTATION_STATUS=$run_status"
if {[string first "Complete" $run_status] < 0} {
    error "Implementation failed: $run_status"
}
set timing_report [file join $project_dir ${project_name}.runs impl_1 system_timing_summary_routed.rpt]
if {![file exists $timing_report]} {
    error "Missing routed timing report: $timing_report"
}
set fp [open $timing_report r]
set timing_text [read $fp]
close $fp
if {[string first "All user specified timing constraints are met." $timing_text] < 0} {
    error "Routed timing constraints are not met; refusing to publish this bitstream"
}

set out_dir [file join $build_dir output]
file mkdir $out_dir
file copy -force \
    [file join $project_dir ${project_name}.runs impl_1 system.bit] \
    [file join $out_dir q6w_gemv.bit]
file copy -force \
    [file join $project_dir ${project_name}.srcs sources_1 bd system hw_handoff system.hwh] \
    [file join $out_dir q6w_gemv.hwh]
puts "BITSTREAM=[file join $out_dir q6w_gemv.bit]"
puts "HWH=[file join $out_dir q6w_gemv.hwh]"
close_project
