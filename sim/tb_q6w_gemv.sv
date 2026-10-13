`timescale 1ns/1ps

module tb_q6w_gemv;
    localparam integer K = 128;
    localparam integer N = 352;
    localparam integer WEIGHT_BYTES = K * 6 / 8;
    localparam integer INPUT_BYTES = K + N * WEIGHT_BYTES;

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

    reg [7:0] input_bytes [0:INPUT_BYTES-1];
    reg [31:0] expected [0:N-1];
    integer received_outputs = 0;
    integer cycle_count = 0;
    integer i;
    integer input_file;
    integer expected_file;

    always #5 clk = ~clk;

    q6w_gemv_axis #(.K(K), .N(N)) dut (
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
            if (received_outputs >= N)
                $fatal(1, "Received too many GEMV outputs");
            if (m_axis_tdata !== expected[received_outputs])
                $fatal(1, "row %0d expected %h, got %h", received_outputs,
                       expected[received_outputs], m_axis_tdata);
            if (m_axis_tlast !== (received_outputs == N - 1))
                $fatal(1, "Incorrect output TLAST at row %0d", received_outputs);
            received_outputs <= received_outputs + 1;
        end
    end

    initial begin
        input_file = $fopen("input.hex", "r");
        if (!input_file)
            $fatal(1, "Cannot open input.hex in the XSim working directory");
        $fclose(input_file);
        expected_file = $fopen("expected.hex", "r");
        if (!expected_file)
            $fatal(1, "Cannot open expected.hex in the XSim working directory");
        $fclose(expected_file);
        $readmemh("input.hex", input_bytes);
        $readmemh("expected.hex", expected);
        if ((^input_bytes[0] === 1'bx) || (^input_bytes[INPUT_BYTES-1] === 1'bx))
            $fatal(1, "Input vector did not load completely");
        if ((^expected[0] === 1'bx) || (^expected[N-1] === 1'bx))
            $fatal(1, "Expected vector did not load completely");
        repeat (5) @(negedge clk);
        rst_n = 1'b1;

        for (i = 0; i < INPUT_BYTES; i = i + 1) begin
            @(negedge clk);
            s_axis_tdata = input_bytes[i];
            s_axis_tlast = (i == INPUT_BYTES - 1);
            s_axis_tvalid = 1'b1;
            @(posedge clk);
            while (!s_axis_tready)
                @(posedge clk);
        end

        @(negedge clk);
        s_axis_tvalid = 1'b0;
        s_axis_tlast = 1'b0;
        while (received_outputs < N)
            @(negedge clk);

        if (protocol_error)
            $fatal(1, "Input TLAST did not match the configured GEMV packet");
        $display("PASS: Q6W GEMV matched all %0d exact integer outputs (%0d input bytes)",
                 N, INPUT_BYTES);
        $finish;
    end

    initial begin
        #20000000;
        $fatal(1, "Timed out: outputs=%0d", received_outputs);
    end
endmodule
