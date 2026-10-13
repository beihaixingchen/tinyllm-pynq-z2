`timescale 1ns/1ps
`default_nettype none

module q6w_gemv_axis #(
    parameter integer K = 128,
    parameter integer N = 352
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire [7:0]        s_axis_tdata,
    input  wire              s_axis_tvalid,
    output wire              s_axis_tready,
    input  wire              s_axis_tlast,
    output reg  [31:0]       m_axis_tdata,
    output reg               m_axis_tvalid,
    input  wire              m_axis_tready,
    output reg               m_axis_tlast,
    output reg               protocol_error
);

    localparam integer GROUPS = K / 4;
    localparam integer X_INDEX_W = (K <= 1) ? 1 : $clog2(K);
    localparam integer ROW_INDEX_W = (N <= 1) ? 1 : $clog2(N);
    localparam integer GROUP_INDEX_W = (GROUPS <= 1) ? 1 : $clog2(GROUPS);

    localparam [1:0] ST_LOAD_X = 2'd0;
    localparam [1:0] ST_LOAD_Q = 2'd1;
    localparam [1:0] ST_MAC    = 2'd2;
    localparam [1:0] ST_OUTPUT = 2'd3;

    reg [1:0] state;
    reg signed [7:0] activation [0:K-1];
    reg [X_INDEX_W-1:0] activation_count;
    reg [ROW_INDEX_W-1:0] row_index;
    reg [GROUP_INDEX_W-1:0] group_index;
    reg [1:0] qbyte_index;
    reg [1:0] mac_lane;
    reg [7:0] packed_byte0;
    reg [7:0] packed_byte1;
    reg signed [5:0] q0, q1, q2, q3;
    reg signed [31:0] accumulator;

    wire input_fire = s_axis_tvalid && s_axis_tready;
    wire input_last_expected = (state == ST_LOAD_Q)
                             && (row_index == N - 1)
                             && (group_index == GROUPS - 1)
                             && (qbyte_index == 2'd2);
    wire [X_INDEX_W-1:0] group_base = group_index;
    wire [X_INDEX_W-1:0] activation_index = (group_base << 2) + mac_lane;

    reg signed [5:0] current_q;
    always @* begin
        case (mac_lane)
            2'd0: current_q = q0;
            2'd1: current_q = q1;
            2'd2: current_q = q2;
            default: current_q = q3;
        endcase
    end

    wire signed [7:0] current_activation = activation[activation_index];
    wire signed [13:0] product = current_q * current_activation;
    wire signed [31:0] product_extended = {{18{product[13]}}, product};
    wire signed [31:0] accumulator_next = accumulator + product_extended;

    assign s_axis_tready = (state == ST_LOAD_X) || (state == ST_LOAD_Q);

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_LOAD_X;
            activation_count <= {X_INDEX_W{1'b0}};
            row_index <= {ROW_INDEX_W{1'b0}};
            group_index <= {GROUP_INDEX_W{1'b0}};
            qbyte_index <= 2'd0;
            mac_lane <= 2'd0;
            packed_byte0 <= 8'd0;
            packed_byte1 <= 8'd0;
            q0 <= 6'sd0;
            q1 <= 6'sd0;
            q2 <= 6'sd0;
            q3 <= 6'sd0;
            accumulator <= 32'sd0;
            m_axis_tdata <= 32'd0;
            m_axis_tvalid <= 1'b0;
            m_axis_tlast <= 1'b0;
            protocol_error <= 1'b0;
        end else begin
            if (input_fire) begin
                if (s_axis_tlast != input_last_expected)
                    protocol_error <= 1'b1;

                if (state == ST_LOAD_X) begin
                    activation[activation_count] <= $signed(s_axis_tdata);
                    if (activation_count == K - 1) begin
                        activation_count <= {X_INDEX_W{1'b0}};
                        row_index <= {ROW_INDEX_W{1'b0}};
                        group_index <= {GROUP_INDEX_W{1'b0}};
                        qbyte_index <= 2'd0;
                        mac_lane <= 2'd0;
                        accumulator <= 32'sd0;
                        state <= ST_LOAD_Q;
                    end else begin
                        activation_count <= activation_count + 1'b1;
                    end
                end else begin
                    case (qbyte_index)
                        2'd0: begin
                            packed_byte0 <= s_axis_tdata;
                            qbyte_index <= 2'd1;
                        end
                        2'd1: begin
                            packed_byte1 <= s_axis_tdata;
                            qbyte_index <= 2'd2;
                        end
                        default: begin
                            q0 <= $signed(packed_byte0[5:0]);
                            q1 <= $signed({packed_byte1[3:0], packed_byte0[7:6]});
                            q2 <= $signed({s_axis_tdata[1:0], packed_byte1[7:4]});
                            q3 <= $signed(s_axis_tdata[7:2]);
                            qbyte_index <= 2'd0;
                            mac_lane <= 2'd0;
                            state <= ST_MAC;
                        end
                    endcase
                end
            end

            if (state == ST_MAC) begin
                accumulator <= accumulator_next;
                if (mac_lane == 2'd3) begin
                    mac_lane <= 2'd0;
                    if (group_index == GROUPS - 1) begin
                        m_axis_tdata <= accumulator_next;
                        m_axis_tvalid <= 1'b1;
                        m_axis_tlast <= (row_index == N - 1);
                        state <= ST_OUTPUT;
                    end else begin
                        group_index <= group_index + 1'b1;
                        state <= ST_LOAD_Q;
                    end
                end else begin
                    mac_lane <= mac_lane + 1'b1;
                end
            end

            if ((state == ST_OUTPUT) && m_axis_tready) begin
                m_axis_tvalid <= 1'b0;
                m_axis_tlast <= 1'b0;
                accumulator <= 32'sd0;
                group_index <= {GROUP_INDEX_W{1'b0}};
                qbyte_index <= 2'd0;
                if (row_index == N - 1) begin
                    row_index <= {ROW_INDEX_W{1'b0}};
                    activation_count <= {X_INDEX_W{1'b0}};
                    state <= ST_LOAD_X;
                end else begin
                    row_index <= row_index + 1'b1;
                    state <= ST_LOAD_Q;
                end
            end
        end
    end

endmodule

`default_nettype wire
