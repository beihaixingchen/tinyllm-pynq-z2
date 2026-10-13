set script_dir [file dirname [file normalize [info script]]]
set project_root [file normalize [file join $script_dir ..]]
set project_dir [file join $project_root build clock_ip_probe]

create_project -force clock_ip_probe $project_dir -part xc7z020clg400-1
create_bd_design clock_probe

create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 clk_wiz
set_property -dict [list \
    CONFIG.PRIM_IN_FREQ {125.000} \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {133.333} \
    CONFIG.USE_LOCKED {true} \
    CONFIG.RESET_TYPE {ACTIVE_HIGH}] [get_bd_cells clk_wiz]

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 axi_cc_mem
set_property -dict [list \
    CONFIG.PROTOCOL {AXI4} \
    CONFIG.READ_WRITE_MODE {READ_WRITE} \
    CONFIG.ADDR_WIDTH {32} \
    CONFIG.DATA_WIDTH {64} \
    CONFIG.ID_WIDTH {1} \
    CONFIG.ACLK_ASYNC {1} \
    CONFIG.SYNCHRONIZATION_STAGES {3}] [get_bd_cells axi_cc_mem]

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 axi_cc_ctrl
set_property -dict [list \
    CONFIG.PROTOCOL {AXI4LITE} \
    CONFIG.READ_WRITE_MODE {READ_WRITE} \
    CONFIG.ADDR_WIDTH {7} \
    CONFIG.DATA_WIDTH {32} \
    CONFIG.ID_WIDTH {0} \
    CONFIG.ACLK_ASYNC {1} \
    CONFIG.SYNCHRONIZATION_STAGES {3}] [get_bd_cells axi_cc_ctrl]

foreach cell [list [get_bd_cells clk_wiz] [get_bd_cells axi_cc_mem] [get_bd_cells axi_cc_ctrl]] {
    puts "IP_CELL=[get_property NAME $cell]"
    foreach property [lsort [list_property $cell]] {
        if {[string match "CONFIG.*" $property]} {
            puts "$property=[get_property $property $cell]"
        }
    }
    foreach pin [lsort [get_bd_pins -of_objects $cell]] {
        puts "PIN=[get_property NAME $pin]"
    }
    foreach intf [lsort [get_bd_intf_pins -of_objects $cell]] {
        puts "INTF=[get_property NAME $intf]"
    }
}

validate_bd_design
save_bd_design
exit
