`timescale 1ns/1ps

module tb_q6w_gemv_batched;
    localparam integer MAX_K = 352;
    localparam integer MAX_ROWS = 4096;
    localparam integer MAX_BATCH = 16;
    localparam integer MAX_INPUT_BYTES = 450000;

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

    reg [7:0] input_bytes [0:MAX_INPUT_BYTES-1];
    reg [31:0] expected [0:MAX_ROWS-1];
    integer batch_count;
    integer matrix_count;
    integer total_outputs;
    integer input_byte_count;
    integer cursor;
    integer matrix_k;
    integer matrix_rows;
    integer received_outputs = 0;
    integer cycle_count = 0;
    integer i;
    integer input_file;
    integer expected_file;

    always #5 clk = ~clk;

    q6w_gemv_batched_axis #(
        .MAX_K(MAX_K),
        .MAX_ROWS(MAX_ROWS),
        .MAX_BATCH(MAX_BATCH)
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
            if (received_outputs >= total_outputs)
                $fatal(1, "Received too many batched GEMV outputs");
            if (m_axis_tdata !== expected[received_outputs])
                $fatal(1, "output %0d expected %h, got %h", received_outputs,
                       expected[received_outputs], m_axis_tdata);
            if (m_axis_tlast !== (received_outputs == total_outputs - 1))
                $fatal(1, "Incorrect output TLAST at output %0d", received_outputs);
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
        if ((^input_bytes[0] === 1'bx) || (^input_bytes[1] === 1'bx))
            $fatal(1, "Batch header did not load completely");
        batch_count = {input_bytes[1], input_bytes[0]};
        if ((batch_count < 1) || (batch_count > MAX_BATCH))
            $fatal(1, "Invalid batch count %0d", batch_count);

        cursor = 2;
        total_outputs = 0;
        for (matrix_count = 0; matrix_count < batch_count; matrix_count = matrix_count + 1) begin
            if ((^input_bytes[cursor+3] === 1'bx))
                $fatal(1, "Matrix %0d header did not load", matrix_count);
            matrix_k = {input_bytes[cursor+1], input_bytes[cursor]};
            matrix_rows = {input_bytes[cursor+3], input_bytes[cursor+2]};
            if ((matrix_k < 4) || (matrix_k > MAX_K) || ((matrix_k % 4) != 0)
                || (matrix_rows < 1) || (matrix_rows > MAX_ROWS))
                $fatal(1, "Invalid matrix %0d shape K=%0d rows=%0d", matrix_count, matrix_k, matrix_rows);
            cursor = cursor + 4 + matrix_k * 2 + matrix_rows * matrix_k * 6 / 8;
            total_outputs = total_outputs + matrix_rows;
            if (total_outputs > MAX_ROWS)
                $fatal(1, "Batch output count exceeds testbench capacity");
        end
        input_byte_count = cursor;
        if (input_byte_count > MAX_INPUT_BYTES)
            $fatal(1, "Batch input exceeds testbench capacity: %0d", input_byte_count);
        if ((^input_bytes[input_byte_count-1] === 1'bx))
            $fatal(1, "Final batch input byte did not load");
        if ((^expected[0] === 1'bx) || (^expected[total_outputs-1] === 1'bx))
            $fatal(1, "Expected outputs did not load completely");

        repeat (5) @(negedge clk);
        rst_n = 1'b1;
        for (i = 0; i < input_byte_count; i = i + 1) begin
            @(negedge clk);
            s_axis_tdata = input_bytes[i];
            s_axis_tlast = (i == input_byte_count - 1);
            s_axis_tvalid = 1'b1;
            @(posedge clk);
            while (!s_axis_tready)
                @(posedge clk);
        end

        @(negedge clk);
        s_axis_tvalid = 1'b0;
        s_axis_tlast = 1'b0;
        while (received_outputs < total_outputs)
            @(negedge clk);

        if (protocol_error)
            $fatal(1, "Input TLAST or batched matrix header protocol error");
        $display("PASS: batched Q6W GEMV operations=%0d outputs=%0d input_bytes=%0d",
                 batch_count, total_outputs, input_byte_count);
        $finish;
    end

    initial begin
        #30000000;
        $fatal(1, "Timed out: outputs=%0d of %0d", received_outputs, total_outputs);
    end
endmodule
