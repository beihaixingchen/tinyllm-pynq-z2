set script_dir [file dirname [file normalize [info script]]]
set project_dir [file normalize [file join $script_dir ..]]
set build_dir [file join $project_dir build autonomous_overlay]
set hls_project tinyllm_autonomous_i8_block2
if {[info exists ::env(AUTONOMOUS_HLS_PROJECT)]} {
    set hls_project $::env(AUTONOMOUS_HLS_PROJECT)
}
set hls_ip_dir [file join $project_dir build autonomous_hls $hls_project solution_z7020 impl ip]
set target_mhz 50
if {[info exists ::env(AUTONOMOUS_PL_MHZ)]} {
    set target_mhz $::env(AUTONOMOUS_PL_MHZ)
}
if {![string is integer -strict $target_mhz] || $target_mhz < 40 || $target_mhz > 150} {
    error "Autonomous PL clock must be an integer from 40 through 150 MHz"
}
set core_mhz ""
if {[info exists ::env(AUTONOMOUS_CORE_MHZ)]} {
    set core_mhz $::env(AUTONOMOUS_CORE_MHZ)
    if {![string is double -strict $core_mhz] || $core_mhz <= $target_mhz || $core_mhz > 180} {
        error "Autonomous core clock must be numeric, above the PS clock, and at most 180 MHz"
    }
}
set memory_fast_clock 0
if {[info exists ::env(AUTONOMOUS_MEMORY_FAST_CLOCK)]} {
    if {$::env(AUTONOMOUS_MEMORY_FAST_CLOCK) ni {0 1}} {
        error "AUTONOMOUS_MEMORY_FAST_CLOCK must be 0 or 1"
    }
    set memory_fast_clock $::env(AUTONOMOUS_MEMORY_FAST_CLOCK)
    if {$memory_fast_clock && $core_mhz eq ""} {
        error "AUTONOMOUS_MEMORY_FAST_CLOCK requires AUTONOMOUS_CORE_MHZ"
    }
}
if {![file exists [file join $hls_ip_dir component.xml]]} {
    error "Missing exported HLS IP at $hls_ip_dir; run build_autonomous_hls.tcl first"
}

set project_name tinyllm_autonomous_z7020
if {[info exists ::env(AUTONOMOUS_OVERLAY_TAG)]} {
    set overlay_tag $::env(AUTONOMOUS_OVERLAY_TAG)
    if {![regexp {^[A-Za-z0-9_]+$} $overlay_tag]} {
        error "AUTONOMOUS_OVERLAY_TAG must contain only letters, digits, and underscores"
    }
    set build_dir [file join $project_dir build "autonomous_overlay_$overlay_tag"]
    append project_name "_$overlay_tag"
}
set project_path [file join $build_dir vivado]
file mkdir $build_dir
file delete -force $project_path
create_project $project_name $project_path -part xc7z020clg400-1 -force
set_property board_part tul.com.tw:pynq-z2:part0:1.0 [current_project]
set_property source_mgmt_mode None [current_project]
set_property ip_repo_paths [list $hls_ip_dir] [current_project]
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
    CONFIG.PCW_USE_S_AXI_HP2 {1} \
    CONFIG.PCW_EN_CLK0_PORT {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ $target_mhz \
    CONFIG.PCW_EN_RST0_PORT {1}] [get_bd_cells processing_system7_0]

create_bd_cell -type ip -vlnv llm-pynq.org:user:tinyllm_autonomous:1.0 tinyllm_autonomous_0
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0

apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/processing_system7_0/M_AXI_GP0" Clk_master "Auto" Clk_slave "Auto"} \
    [get_bd_intf_pins tinyllm_autonomous_0/s_axi_control]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/tinyllm_autonomous_0/m_axi_gmem0" Clk_master "Auto" Clk_slave "Auto"} \
    [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/tinyllm_autonomous_0/m_axi_gmem1" Clk_master "Auto" Clk_slave "Auto"} \
    [get_bd_intf_pins processing_system7_0/S_AXI_HP1]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/tinyllm_autonomous_0/m_axi_gmem2" Clk_master "Auto" Clk_slave "Auto"} \
    [get_bd_intf_pins processing_system7_0/S_AXI_HP2]
set gmem3_pin [get_bd_intf_pins -quiet tinyllm_autonomous_0/m_axi_gmem3]
if {[llength $gmem3_pin] > 0} {
    set_property CONFIG.PCW_USE_S_AXI_HP3 {1} [get_bd_cells processing_system7_0]
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
        -config {Master "/tinyllm_autonomous_0/m_axi_gmem3" Clk_master "Auto" Clk_slave "Auto"} \
        [get_bd_intf_pins processing_system7_0/S_AXI_HP3]
}

if {[info exists ::env(AUTONOMOUS_HP0_REGSLICE)] && $::env(AUTONOMOUS_HP0_REGSLICE) eq "1"} {
    set hp0_interconnect [get_bd_cells axi_mem_intercon]
    if {[llength $hp0_interconnect] != 1} {
        error "Expected one AXI interconnect between gmem0 and HP0"
    }
    set_property CONFIG.M00_HAS_REGSLICE {1} $hp0_interconnect
    puts "HP0_OUTPUT_REGSLICE=enabled"
}

if {$core_mhz ne ""} {
    set slow_mhz [get_property CONFIG.PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ \
        [get_bd_cells processing_system7_0]]

    create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 pl_clk_wiz
    set_property -dict [list \
        CONFIG.PRIM_IN_FREQ [format "%.3f" $slow_mhz] \
        CONFIG.CLKOUT1_REQUESTED_OUT_FREQ [format "%.3f" $core_mhz] \
        CONFIG.USE_LOCKED {true} \
        CONFIG.RESET_TYPE {ACTIVE_HIGH}] [get_bd_cells pl_clk_wiz]
    connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
        [get_bd_pins pl_clk_wiz/clk_in1]
    connect_bd_net [get_bd_pins rst_ps7_0/peripheral_reset] \
        [get_bd_pins pl_clk_wiz/reset]

    create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_pl_fast
    connect_bd_net [get_bd_pins pl_clk_wiz/clk_out1] \
        [get_bd_pins rst_pl_fast/slowest_sync_clk]
    connect_bd_net [get_bd_pins pl_clk_wiz/locked] \
        [get_bd_pins rst_pl_fast/dcm_locked]
    connect_bd_net [get_bd_pins rst_ps7_0/peripheral_reset] \
        [get_bd_pins rst_pl_fast/ext_reset_in]

    set links [list \
        [list control tinyllm_autonomous_0/s_axi_control axi_cc_control control AXI4LITE 7 32 0]]
    if {!$memory_fast_clock} {
        lappend links \
            [list gmem0 tinyllm_autonomous_0/m_axi_gmem0 axi_cc_gmem0 memory AXI4 32 64 1] \
            [list gmem1 tinyllm_autonomous_0/m_axi_gmem1 axi_cc_gmem1 memory AXI4 32 32 1] \
            [list gmem2 tinyllm_autonomous_0/m_axi_gmem2 axi_cc_gmem2 memory AXI4 32 64 1]
    }
    if {[llength [get_bd_intf_pins -quiet tinyllm_autonomous_0/m_axi_gmem3]] > 0} {
        if {!$memory_fast_clock} {
            lappend links [list gmem3 tinyllm_autonomous_0/m_axi_gmem3 axi_cc_gmem3 memory AXI4 32 64 1]
        }
    }

    foreach link $links {
        lassign $link link_name hls_intf converter_name role protocol addr_width data_width id_width
        create_bd_cell -type ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 $converter_name
        set_property -dict [list \
            CONFIG.PROTOCOL $protocol \
            CONFIG.READ_WRITE_MODE {READ_WRITE} \
            CONFIG.ADDR_WIDTH $addr_width \
            CONFIG.DATA_WIDTH $data_width \
            CONFIG.ID_WIDTH $id_width \
            CONFIG.ACLK_ASYNC {1} \
            CONFIG.SYNCHRONIZATION_STAGES {3}] [get_bd_cells $converter_name]

        set hls_pin [get_bd_intf_pins $hls_intf]
        set net [get_bd_intf_nets -of_objects $hls_pin]
        set peers [get_bd_intf_pins -of_objects $net]
        set peer ""
        foreach candidate $peers {
            if {[get_property NAME $candidate] ne [get_property NAME $hls_pin]} {
                set peer $candidate
            }
        }
        if {$peer eq ""} {
            error "Could not find AXI automation peer for $hls_intf"
        }
        disconnect_bd_intf_net $net $hls_pin
        set peer_net [get_bd_intf_nets -of_objects $peer]
        if {[llength $peer_net] > 0} {
            disconnect_bd_intf_net $peer_net $peer
        }

        if {$role eq "memory"} {
            connect_bd_intf_net $hls_pin [get_bd_intf_pins $converter_name/S_AXI]
            connect_bd_intf_net [get_bd_intf_pins $converter_name/M_AXI] $peer
            set fast_aclk_pin $converter_name/s_axi_aclk
            set slow_aclk_pin $converter_name/m_axi_aclk
            set fast_reset_pin $converter_name/s_axi_aresetn
            set slow_reset_pin $converter_name/m_axi_aresetn
        } else {
            connect_bd_intf_net $peer [get_bd_intf_pins $converter_name/S_AXI]
            connect_bd_intf_net [get_bd_intf_pins $converter_name/M_AXI] $hls_pin
            set slow_aclk_pin $converter_name/s_axi_aclk
            set fast_aclk_pin $converter_name/m_axi_aclk
            set slow_reset_pin $converter_name/s_axi_aresetn
            set fast_reset_pin $converter_name/m_axi_aresetn
        }
        connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
            [get_bd_pins $slow_aclk_pin]
        connect_bd_net [get_bd_pins pl_clk_wiz/clk_out1] \
            [get_bd_pins $fast_aclk_pin]
        connect_bd_net [get_bd_pins rst_ps7_0/peripheral_aresetn] \
            [get_bd_pins $slow_reset_pin]
        connect_bd_net [get_bd_pins rst_pl_fast/peripheral_aresetn] \
            [get_bd_pins $fast_reset_pin]
        puts "ASYNC_AXI_CONVERTER=$link_name"
    }

    if {$memory_fast_clock} {
        set memory_clock_pins [list]
        set memory_reset_pins [list]
        foreach memory_link [list \
            [list gmem0 tinyllm_autonomous_0/m_axi_gmem0 0] \
            [list gmem1 tinyllm_autonomous_0/m_axi_gmem1 1] \
            [list gmem2 tinyllm_autonomous_0/m_axi_gmem2 2]] {
            lassign $memory_link link_name hls_intf hp_index
            set hls_pin [get_bd_intf_pins $hls_intf]
            set net [get_bd_intf_nets -of_objects $hls_pin]
            set peers [get_bd_intf_pins -of_objects $net]
            set interconnect ""
            foreach candidate $peers {
                if {[get_property NAME $candidate] ne [get_property NAME $hls_pin]} {
                    set interconnect [get_bd_cells -of_objects $candidate]
                }
            }
            if {$interconnect eq ""} {
                error "Could not find memory interconnect for $link_name"
            }
            foreach suffix {ACLK S00_ACLK M00_ACLK} {
                lappend memory_clock_pins [get_bd_pins $interconnect/$suffix]
            }
            foreach suffix {ARESETN S00_ARESETN M00_ARESETN} {
                lappend memory_reset_pins [get_bd_pins $interconnect/$suffix]
            }
            lappend memory_clock_pins \
                [get_bd_pins processing_system7_0/S_AXI_HP${hp_index}_ACLK]
        }
        set gmem3_hls [get_bd_intf_pins -quiet tinyllm_autonomous_0/m_axi_gmem3]
        if {[llength $gmem3_hls] > 0} {
            set net [get_bd_intf_nets -of_objects $gmem3_hls]
            foreach candidate [get_bd_intf_pins -of_objects $net] {
                if {[get_property NAME $candidate] ne [get_property NAME $gmem3_hls]} {
                    set interconnect [get_bd_cells -of_objects $candidate]
                }
            }
            foreach suffix {ACLK S00_ACLK M00_ACLK} {
                lappend memory_clock_pins [get_bd_pins $interconnect/$suffix]
            }
            foreach suffix {ARESETN S00_ARESETN M00_ARESETN} {
                lappend memory_reset_pins [get_bd_pins $interconnect/$suffix]
            }
            lappend memory_clock_pins [get_bd_pins processing_system7_0/S_AXI_HP3_ACLK]
        }
        foreach pin $memory_clock_pins {
            set net [get_bd_nets -of_objects $pin]
            if {[llength $net] > 0} {
                disconnect_bd_net $net $pin
            }
        }
        foreach pin $memory_reset_pins {
            set net [get_bd_nets -of_objects $pin]
            if {[llength $net] > 0} {
                disconnect_bd_net $net $pin
            }
        }
        connect_bd_net [get_bd_pins pl_clk_wiz/clk_out1] {*}$memory_clock_pins
        connect_bd_net [get_bd_pins rst_pl_fast/peripheral_aresetn] {*}$memory_reset_pins
        puts "MEMORY_INTERCONNECT_AND_HP_CLOCK=fast"
    }

    set ap_clk [get_bd_pins tinyllm_autonomous_0/ap_clk]
    disconnect_bd_net [get_bd_nets -of_objects $ap_clk] $ap_clk
    connect_bd_net [get_bd_pins pl_clk_wiz/clk_out1] $ap_clk
    set ap_reset [get_bd_pins tinyllm_autonomous_0/ap_rst_n]
    disconnect_bd_net [get_bd_nets -of_objects $ap_reset] $ap_reset
    connect_bd_net [get_bd_pins rst_pl_fast/peripheral_aresetn] $ap_reset
    puts "PS_AXI_CLOCK_MHZ=$slow_mhz CORE_CLOCK_REQUEST_MHZ=$core_mhz"
}

if {![llength [get_bd_nets -of_objects [get_bd_pins tinyllm_autonomous_0/ap_clk]]]} {
    error "AXI automation did not connect the HLS clock"
}
if {![llength [get_bd_nets -of_objects [get_bd_pins tinyllm_autonomous_0/ap_rst_n]]]} {
    error "AXI automation did not connect the HLS reset"
}

assign_bd_address
validate_bd_design
save_bd_design

if {[info exists ::env(AUTONOMOUS_BD_ONLY)] && $::env(AUTONOMOUS_BD_ONLY) eq "1"} {
    puts "BLOCK_DESIGN_VALIDATION=passed"
    exit
}

set actual_mhz [get_property CONFIG.PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ \
    [get_bd_cells processing_system7_0]]
if {![string is double -strict $actual_mhz] || $actual_mhz <= 0} {
    error "Could not determine actual PS7 FCLK0 frequency: $actual_mhz"
}
set bd_file [get_files [file join $project_path ${project_name}.srcs sources_1 bd system system.bd]]
generate_target all $bd_file

set ps_xdc [file join $project_path ${project_name}.srcs sources_1 bd system ip \
    system_processing_system7_0_0 system_processing_system7_0_0.xdc]
if {![file exists $ps_xdc]} {
    error "Missing generated PS7 clock constraints: $ps_xdc"
}
set fp [open $ps_xdc r]
set xdc_text [read $fp]
close $fp
set clock_pattern {create_clock -name clk_fpga_0 -period "[0-9]+(\.[0-9]+)?"}
set clock_period [expr {1000.0 / $actual_mhz}]
set new_clock [format {create_clock -name clk_fpga_0 -period "%.6f"} $clock_period]
if {![regexp $clock_pattern $xdc_text old_clock]} {
    error "Could not find generated clk_fpga_0 constraint in $ps_xdc"
}
set xdc_text [string map [list $old_clock $new_clock] $xdc_text]
set fp [open $ps_xdc w]
puts -nonewline $fp $xdc_text
close $fp
puts "PS7_ACTUAL_CLOCK_MHZ=$actual_mhz"
puts "PS7_CLOCK_CONSTRAINT_NS=$clock_period"

set source_set [get_filesets sources_1]
set_property top_auto_set false $source_set
set_property top system $source_set
update_compile_order -fileset sources_1
if {[info exists ::env(AUTONOMOUS_PHYS_OPT_DIRECTIVE)]} {
    set directive $::env(AUTONOMOUS_PHYS_OPT_DIRECTIVE)
    set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
    set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE $directive [get_runs impl_1]
    puts "PHYS_OPT_DIRECTIVE=$directive"
}
if {[info exists ::env(AUTONOMOUS_FORCE_REPLICATE_NET)]} {
    set force_replicate_script [file join $script_dir force_replicate_net.tcl]
    if {![file exists $force_replicate_script]} {
        error "Missing forced replication script: $force_replicate_script"
    }
    set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
    set_property STEPS.PHYS_OPT_DESIGN.TCL.POST $force_replicate_script [get_runs impl_1]
    puts "PHYS_OPT_FORCE_REPLICATE_NET=$::env(AUTONOMOUS_FORCE_REPLICATE_NET)"
}
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
set run_status [get_property STATUS [get_runs impl_1]]
puts "IMPLEMENTATION_STATUS=$run_status"
if {[string first "Complete" $run_status] < 0} {
    error "Implementation failed: $run_status"
}

set timing_report [file join $project_path ${project_name}.runs impl_1 system_timing_summary_routed.rpt]
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
file copy -force [file join $project_path ${project_name}.runs impl_1 system.bit] \
    [file join $out_dir tinyllm_autonomous.bit]
file copy -force [file join $project_path ${project_name}.srcs sources_1 bd system hw_handoff system.hwh] \
    [file join $out_dir tinyllm_autonomous.hwh]
puts "BITSTREAM=[file join $out_dir tinyllm_autonomous.bit]"
puts "HWH=[file join $out_dir tinyllm_autonomous.hwh]"
close_project
