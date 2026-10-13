set script_dir [file dirname [file normalize [info script]]]
set demo_dir [file normalize [file join $script_dir ..]]
set rtl_file [file join $demo_dir rtl q6w_gemv_dynamic_axis.sv]
set build_dir [file join $demo_dir build all_linears]
set project_dir [file join $build_dir vivado]
set ip_project_dir [file join $build_dir ip_packager]
set ip_repo_dir [file join $build_dir ip_repo]
set project_name q6w_gemv_all

file delete -force $ip_project_dir $ip_repo_dir
create_project q6w_gemv_all_ip $ip_project_dir -part xc7z020clg400-1 -force
set_property source_mgmt_mode None [current_project]
add_files -norecurse $rtl_file
set rtl_obj [get_files $rtl_file]
set_property file_type Verilog $rtl_obj
set_property top q6w_gemv_dynamic_axis [get_filesets sources_1]
update_compile_order -fileset sources_1

ipx::package_project -root_dir $ip_repo_dir -vendor llm-pynq.org \
    -library user -taxonomy /UserIP -import_files -set_current true
set core [ipx::current_core]
set_property name q6w_gemv_dynamic_axis $core
set_property version 1.0 $core
set_property display_name {Q6W Dynamic AXI Stream GEMV} $core
set_property description {Runtime-sized signed int6 by int8 GEMV, maximum 352 by 352} $core
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
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
    CONFIG.PCW_EN_RST0_PORT {1}] [get_bd_cells processing_system7_0]

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_include_mm2s {1} \
    CONFIG.c_include_s2mm {1} \
    CONFIG.c_sg_length_width {17} \
    CONFIG.c_m_axis_mm2s_tdata_width {8} \
    CONFIG.c_mm2s_burst_size {16} \
    CONFIG.c_s2mm_burst_size {16}] [get_bd_cells axi_dma_0]

create_bd_cell -type ip -vlnv llm-pynq.org:user:q6w_gemv_dynamic_axis:1.0 q6w_gemv_0
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_100M

apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/processing_system7_0/M_AXI_GP0" Clk_master "Auto" Clk_slave "Auto"} \
    [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/axi_dma_0/M_AXI_MM2S" Clk_master "Auto" Clk_slave "Auto"} \
    [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/axi_dma_0/M_AXI_S2MM" Clk_master "Auto" Clk_slave "Auto"} \
    [get_bd_intf_pins processing_system7_0/S_AXI_HP1]

connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
    [get_bd_pins rst_ps7_100M/slowest_sync_clk] \
    [get_bd_pins axi_dma_0/s_axi_lite_aclk] \
    [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] \
    [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] \
    [get_bd_pins q6w_gemv_0/clk]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] \
    [get_bd_pins axi_dma_0/axi_resetn] \
    [get_bd_pins q6w_gemv_0/rst_n]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] \
    [get_bd_intf_pins q6w_gemv_0/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins q6w_gemv_0/M_AXIS] \
    [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

assign_bd_address
validate_bd_design
save_bd_design

set bd_file [get_files [file join $project_dir ${project_name}.srcs sources_1 bd system system.bd]]
generate_target all $bd_file
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
