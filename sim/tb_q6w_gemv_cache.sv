`timescale 1ns/1ps

module tb_q6w_gemv_cache;
    localparam integer MAX_INPUT_BYTES = 500000;
    localparam integer MAX_OUTPUTS = 4800;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg [7:0] s_axis_tdata = 8'd0;
    reg s_axis_tvalid = 1'b0;
    wire s_axis_tready;
    reg s_axis_tlast = 1'b0;
    wire [31:0] m_axis_tdata;
    wire m_axis_tvalid;
    reg m_axis_tready = 1'b0;
    wire m_axis_tlast;
    wire protocol_error;

    reg [7:0] load_bytes [0:MAX_INPUT_BYTES-1];
    reg [7:0] compute_bytes [0:4095];
    reg [31:0] expected [0:MAX_OUTPUTS-1];
    integer load_count;
    integer compute_count;
    integer load_cursor;
    integer compute_cursor;
    integer load_ops;
    integer compute_ops;
    integer slot;
    integer load_packet_bytes;
    integer expected_outputs;
    integer received = 0;
    integer cycle_count = 0;
    integer compute_start_cycle = 0;
    integer compute_end_cycle = 0;
    integer i;
    integer load_file;
    integer compute_file;
    integer expected_file;
    reg compute_started = 1'b0;

    function integer rows_for_slot;
        input integer cache_slot;
        begin
            rows_for_slot = (cache_slot >= 10) ? 256 : 352;
        end
    endfunction

    always #5 clk = ~clk;

    q6w_gemv_batched_axis #(
        .MAX_K(352),
        .MAX_ROWS(4096),
        .MAX_BATCH(32)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_axis_tlast),
        .protocol_error(protocol_error)
    );

    always @(negedge clk) begin
        cycle_count <= cycle_count + 1;
        m_axis_tready <= (cycle_count % 11 != 0) && (cycle_count % 17 != 0);
    end

    always @(posedge clk) begin
        if (rst_n && m_axis_tvalid && m_axis_tready) begin
            if (!compute_started)
                $fatal(1, "Cache preload unexpectedly produced output");
            if (received >= MAX_OUTPUTS)
                $fatal(1, "Received too many cached GEMV outputs");
            if (m_axis_tdata !== expected[received])
                $fatal(1, "cached output %0d expected %h, got %h", received,
                       expected[received], m_axis_tdata);
            if (m_axis_tlast !== (received == expected_outputs - 1))
                $fatal(1, "Incorrect cached output TLAST at output %0d", received);
            received <= received + 1;
        end
    end

    initial begin
        load_file = $fopen("load.hex", "r");
        if (!load_file)
            $fatal(1, "Cannot open load.hex in the XSim working directory");
        $fclose(load_file);
        compute_file = $fopen("compute.hex", "r");
        if (!compute_file)
            $fatal(1, "Cannot open compute.hex in the XSim working directory");
        $fclose(compute_file);
        expected_file = $fopen("expected.hex", "r");
        if (!expected_file)
            $fatal(1, "Cannot open expected.hex in the XSim working directory");
        $fclose(expected_file);

        $readmemh("load.hex", load_bytes);
        $readmemh("compute.hex", compute_bytes);
        $readmemh("expected.hex", expected);
        load_ops = {load_bytes[1], load_bytes[0]};
        compute_ops = {compute_bytes[1], compute_bytes[0]};
        if ((load_ops != 10 && load_ops != 15) || load_ops != compute_ops)
            $fatal(1, "Unexpected cache operation counts load=%0d compute=%0d", load_ops, compute_ops);
        expected_outputs = (compute_ops == 15) ? 4800 : 3520;

        load_cursor = 2;
        for (slot = 0; slot < load_ops; slot = slot + 1) begin
            if (({load_bytes[load_cursor+1], load_bytes[load_cursor]} != 0)
                || (load_bytes[load_cursor+2] != slot)
                || (load_bytes[load_cursor+3] != 0))
                $fatal(1, "Invalid cache-load descriptor for slot %0d", slot);
            load_cursor = load_cursor + 4 + rows_for_slot(slot) * 96;
        end
        load_count = load_cursor;
        if (load_count > MAX_INPUT_BYTES || (^load_bytes[load_count-1] === 1'bx))
            $fatal(1, "Cache load packet did not load completely");

        compute_cursor = 2;
        for (slot = 0; slot < compute_ops; slot = slot + 1) begin
            if (({compute_bytes[compute_cursor+1], compute_bytes[compute_cursor]} != 16'hffff)
                && !((slot < 10) && (slot % 2)
                     && ({compute_bytes[compute_cursor+1], compute_bytes[compute_cursor]} == 16'hfffe)))
                $fatal(1, "Invalid cached-compute activation mode for slot %0d", slot);
            if ((slot >= 10 && compute_ops != 15)
                || (compute_bytes[compute_cursor+2] != slot)
                || (compute_bytes[compute_cursor+3] != 0))
                $fatal(1, "Invalid cached-compute descriptor for slot %0d", slot);
            compute_cursor = compute_cursor + 4
                + (({compute_bytes[compute_cursor+1], compute_bytes[compute_cursor]} == 16'hfffe) ? 0 : 256);
        end
        compute_count = compute_cursor;
        if ((^compute_bytes[compute_count-1] === 1'bx)
            || (^expected[0] === 1'bx) || (^expected[expected_outputs-1] === 1'bx))
            $fatal(1, "Compute input or expected outputs did not load completely");

        repeat (5) @(negedge clk);
        rst_n = 1'b1;
        load_cursor = 2;
        for (slot = 0; slot < load_ops; slot = slot + 1) begin
            load_packet_bytes = 2 + 4 + rows_for_slot(slot) * 96;
            for (i = 0; i < load_packet_bytes; i = i + 1) begin
                @(negedge clk);
                if (i == 0)
                    s_axis_tdata = 8'd1;
                else if (i == 1)
                    s_axis_tdata = 8'd0;
                else
                    s_axis_tdata = load_bytes[load_cursor + i - 2];
                s_axis_tlast = (i == load_packet_bytes - 1);
                s_axis_tvalid = 1'b1;
                @(posedge clk);
                while (!s_axis_tready)
                    @(posedge clk);
            end
            load_cursor = load_cursor + 4 + rows_for_slot(slot) * 96;
            @(negedge clk);
            s_axis_tvalid = 1'b0;
            s_axis_tlast = 1'b0;
            repeat (2) @(negedge clk);
        end
        @(negedge clk);
        s_axis_tvalid = 1'b0;
        s_axis_tlast = 1'b0;
        repeat (5) @(negedge clk);

        compute_started = 1'b1;
        compute_start_cycle = cycle_count;
        for (i = 0; i < compute_count; i = i + 1) begin
            @(negedge clk);
            s_axis_tdata = compute_bytes[i];
            s_axis_tlast = (i == compute_count - 1);
            s_axis_tvalid = 1'b1;
            @(posedge clk);
            while (!s_axis_tready)
                @(posedge clk);
        end
        @(negedge clk);
        s_axis_tvalid = 1'b0;
        s_axis_tlast = 1'b0;
        while (received < expected_outputs)
            @(negedge clk);
        compute_end_cycle = cycle_count;

        if (protocol_error)
            $fatal(1, "Cache stream TLAST, slot, or matrix protocol error");
        $display("PASS: cached Q6W GEMV loaded %0d matrix packets (%0d bytes), then matched %0d outputs in %0d cycles",
                 load_ops, load_count, received, compute_end_cycle - compute_start_cycle);
        $finish;
    end

    initial begin
        #50000000;
        $fatal(1, "Timed out: cached outputs=%0d of %0d", received, MAX_OUTPUTS);
    end
endmodule
