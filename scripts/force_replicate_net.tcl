if {![info exists ::env(AUTONOMOUS_FORCE_REPLICATE_NET)]} {
    error "AUTONOMOUS_FORCE_REPLICATE_NET is required"
}

set net_pattern $::env(AUTONOMOUS_FORCE_REPLICATE_NET)
set target_nets [get_nets -hierarchical -filter "NAME =~ $net_pattern"]
if {[llength $target_nets] == 0} {
    set target_pins [get_pins -hierarchical -filter "NAME =~ $net_pattern"]
    if {[llength $target_pins] > 0} {
        set target_nets [get_nets -of_objects $target_pins]
    }
}
if {[llength $target_nets] == 0} {
    error "No nets matched forced replication pattern: $net_pattern"
}

puts "FORCE_REPLICATE_MATCHES=$target_nets"
phys_opt_design -force_replication_on_nets $target_nets
