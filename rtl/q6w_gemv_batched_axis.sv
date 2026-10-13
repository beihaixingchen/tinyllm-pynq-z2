`timescale 1ns/1ps
`default_nettype none

module q6w_gemv_batched_axis #(
    parameter integer MAX_K = 352,
    parameter integer MAX_ROWS = 4096,
    parameter integer MAX_BATCH = 32,
    parameter integer CACHE_SLOTS = 10,
    parameter integer CACHE_WORDS = 112640,
    parameter integer CACHE_MATRIX_ROWS = 352
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [7:0]  s_axis_tdata,
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,
    input  wire        s_axis_tlast,
    output reg  [31:0] m_axis_tdata,
    output reg         m_axis_tvalid,
    input  wire        m_axis_tready,
    output reg         m_axis_tlast,
    output reg         protocol_error
);

    localparam integer K_INDEX_W = (MAX_K <= 1) ? 1 : $clog2(MAX_K);
    localparam integer ROW_INDEX_W = (MAX_ROWS <= 1) ? 1 : $clog2(MAX_ROWS);
    localparam integer GROUP_INDEX_W = (MAX_K <= 4) ? 1 : $clog2(MAX_K / 4);
    localparam integer CACHE_BANK_WORDS = CACHE_WORDS / 4;

    localparam [3:0] ST_BATCH_HEADER = 4'd0;
    localparam [3:0] ST_MATRIX_HEADER = 4'd1;
    localparam [3:0] ST_LOAD_X = 4'd2;
    localparam [3:0] ST_LOAD_Q = 4'd3;
    localparam [3:0] ST_MAC_MULT = 4'd4;
    localparam [3:0] ST_MAC_SUM = 4'd5;
    localparam [3:0] ST_MAC_ACC = 4'd6;
    localparam [3:0] ST_OUTPUT = 4'd7;
    localparam [3:0] ST_CACHE_READ = 4'd8;
    localparam [3:0] ST_CACHE_DECODE = 4'd9;
    localparam [3:0] ST_MAC_MERGE = 4'd10;
    localparam [3:0] ST_MAC_FINAL = 4'd11;

    localparam [1:0] MODE_INLINE = 2'd0;
    localparam [1:0] MODE_CACHE_LOAD = 2'd1;
    localparam [1:0] MODE_CACHE_RUN = 2'd2;

    reg [3:0] state;
    reg batch_header_index;
    reg [7:0] batch_count_lo;
    reg [15:0] batch_count;
    reg [15:0] operation_index;
    reg [1:0] header_index;
    reg [7:0] header_k_lo;
    reg [7:0] header_k_hi;
    reg [7:0] header_rows_lo;
    reg [15:0] active_k;
    reg [15:0] active_rows;
    reg [1:0] active_mode;
    reg [4:0] cache_slot;
    reg [17:0] cache_address;
    reg [CACHE_SLOTS-1:0] cache_valid;
    (* ram_style = "block" *) reg [23:0] q6_cache_bank0 [0:CACHE_BANK_WORDS-1];
    (* ram_style = "block" *) reg [23:0] q6_cache_bank1 [0:CACHE_BANK_WORDS-1];
    (* ram_style = "block" *) reg [23:0] q6_cache_bank2 [0:CACHE_BANK_WORDS-1];
    (* ram_style = "block" *) reg [23:0] q6_cache_bank3 [0:CACHE_BANK_WORDS-1];
    reg [23:0] cache_read_bank0, cache_read_bank1, cache_read_bank2, cache_read_bank3;
    reg signed [15:0] activation_bank0 [0:MAX_K/4-1];
    reg signed [15:0] activation_bank1 [0:MAX_K/4-1];
    reg signed [15:0] activation_bank2 [0:MAX_K/4-1];
    reg signed [15:0] activation_bank3 [0:MAX_K/4-1];
    reg [63:0] activation_stage0, activation_stage1, activation_stage2, activation_stage3;
    reg [63:0] activation_word_bank0 [0:MAX_K/16-1];
    reg [63:0] activation_word_bank1 [0:MAX_K/16-1];
    reg [63:0] activation_word_bank2 [0:MAX_K/16-1];
    reg [63:0] activation_word_bank3 [0:MAX_K/16-1];
    reg [63:0] activation_read_bank0, activation_read_bank1;
    reg [63:0] activation_read_bank2, activation_read_bank3;
    reg [K_INDEX_W-1:0] activation_count;
    reg activation_byte_index;
    reg [7:0] activation_byte_low;
    reg [ROW_INDEX_W-1:0] row_index;
    reg [GROUP_INDEX_W-1:0] group_index;
    reg [GROUP_INDEX_W-1:0] last_group_index;
    reg [1:0] qbyte_index;
    reg [7:0] packed_byte0;
    reg [7:0] packed_byte1;
    (* keep = "true" *) reg signed [5:0] q0, q1, q2, q3, q4, q5, q6, q7;
    (* keep = "true" *) reg signed [5:0] q8, q9, q10, q11, q12, q13, q14, q15;
    reg signed [15:0] x0, x1, x2, x3, x4, x5, x6, x7;
    reg signed [15:0] x8, x9, x10, x11, x12, x13, x14, x15;
    reg signed [21:0] product0_reg, product1_reg, product2_reg, product3_reg;
    reg signed [21:0] product4_reg, product5_reg, product6_reg, product7_reg;
    reg signed [21:0] product8_reg, product9_reg, product10_reg, product11_reg;
    reg signed [21:0] product12_reg, product13_reg, product14_reg, product15_reg;
    reg signed [23:0] quartet0_reg, quartet1_reg, quartet2_reg, quartet3_reg;
    reg signed [23:0] group_low_reg, group_high_reg;
    reg signed [24:0] group_sum_reg;
    reg signed [31:0] accumulator;

    wire input_fire = s_axis_tvalid && s_axis_tready;
    wire [15:0] packet_batch_count = {s_axis_tdata, batch_count_lo};
    wire [15:0] packet_k = {header_k_hi, header_k_lo};
    wire [15:0] packet_rows = {s_axis_tdata, header_rows_lo};
    wire input_last_expected =
        ((state == ST_LOAD_Q)
         && (active_mode != MODE_CACHE_RUN)
         && (operation_index == batch_count - 1'b1)
         && (row_index == active_rows - 1'b1)
         && (group_index == last_group_index)
         && (qbyte_index == 2'd2))
        || ((state == ST_LOAD_X)
            && (active_mode == MODE_CACHE_RUN)
            && activation_byte_index
            && (activation_count == active_k - 1'b1)
            && (operation_index == batch_count - 1'b1));
    function [17:0] cache_base_address;
        input [4:0] slot;
        begin
            if (CACHE_SLOTS == 1) begin
                cache_base_address = 18'd0;
            end else begin
                case (slot)
                    5'd0: cache_base_address = 18'd0;
                    5'd1: cache_base_address = 18'd11264;
                    5'd2: cache_base_address = 18'd22528;
                    5'd3: cache_base_address = 18'd33792;
                    5'd4: cache_base_address = 18'd45056;
                    5'd5: cache_base_address = 18'd56320;
                    5'd6: cache_base_address = 18'd67584;
                    5'd7: cache_base_address = 18'd78848;
                    5'd8: cache_base_address = 18'd90112;
                    5'd9: cache_base_address = 18'd101376;
                    default: cache_base_address = 18'd0;
                endcase
            end
        end
    endfunction
    function [15:0] cache_rows;
        input [4:0] slot;
        begin
            cache_rows = CACHE_MATRIX_ROWS;
        end
    endfunction
    wire signed [21:0] product0_next = q0 * x0;
    wire signed [21:0] product1_next = q1 * x1;
    wire signed [21:0] product2_next = q2 * x2;
    wire signed [21:0] product3_next = q3 * x3;
    wire signed [21:0] product4_next = q4 * x4;
    wire signed [21:0] product5_next = q5 * x5;
    wire signed [21:0] product6_next = q6 * x6;
    wire signed [21:0] product7_next = q7 * x7;
    wire signed [21:0] product8_next = q8 * x8;
    wire signed [21:0] product9_next = q9 * x9;
    wire signed [21:0] product10_next = q10 * x10;
    wire signed [21:0] product11_next = q11 * x11;
    wire signed [21:0] product12_next = q12 * x12;
    wire signed [21:0] product13_next = q13 * x13;
    wire signed [21:0] product14_next = q14 * x14;
    wire signed [21:0] product15_next = q15 * x15;
    wire signed [22:0] pair_sum0_next =
        {{1{product0_reg[21]}}, product0_reg} + {{1{product1_reg[21]}}, product1_reg};
    wire signed [22:0] pair_sum1_next =
        {{1{product2_reg[21]}}, product2_reg} + {{1{product3_reg[21]}}, product3_reg};
    wire signed [23:0] group_sum_next =
        {{1{pair_sum0_next[22]}}, pair_sum0_next} + {{1{pair_sum1_next[22]}}, pair_sum1_next};
    wire signed [22:0] pair_sum2_next =
        {{1{product4_reg[21]}}, product4_reg} + {{1{product5_reg[21]}}, product5_reg};
    wire signed [22:0] pair_sum3_next =
        {{1{product6_reg[21]}}, product6_reg} + {{1{product7_reg[21]}}, product7_reg};
    wire signed [23:0] group_high_next =
        {{1{pair_sum2_next[22]}}, pair_sum2_next} + {{1{pair_sum3_next[22]}}, pair_sum3_next};
    wire signed [24:0] group_total_next =
        {{1{group_low_reg[23]}}, group_low_reg} + {{1{group_high_reg[23]}}, group_high_reg};
    wire signed [22:0] pair_sum4_next =
        {{1{product8_reg[21]}}, product8_reg} + {{1{product9_reg[21]}}, product9_reg};
    wire signed [22:0] pair_sum5_next =
        {{1{product10_reg[21]}}, product10_reg} + {{1{product11_reg[21]}}, product11_reg};
    wire signed [22:0] pair_sum6_next =
        {{1{product12_reg[21]}}, product12_reg} + {{1{product13_reg[21]}}, product13_reg};
    wire signed [22:0] pair_sum7_next =
        {{1{product14_reg[21]}}, product14_reg} + {{1{product15_reg[21]}}, product15_reg};
    wire signed [23:0] quartet0_next =
        {{1{pair_sum0_next[22]}}, pair_sum0_next} + {{1{pair_sum1_next[22]}}, pair_sum1_next};
    wire signed [23:0] quartet1_next =
        {{1{pair_sum2_next[22]}}, pair_sum2_next} + {{1{pair_sum3_next[22]}}, pair_sum3_next};
    wire signed [23:0] quartet2_next =
        {{1{pair_sum4_next[22]}}, pair_sum4_next} + {{1{pair_sum5_next[22]}}, pair_sum5_next};
    wire signed [23:0] quartet3_next =
        {{1{pair_sum6_next[22]}}, pair_sum6_next} + {{1{pair_sum7_next[22]}}, pair_sum7_next};
    wire signed [24:0] cached_low_next =
        {{1{quartet0_reg[23]}}, quartet0_reg} + {{1{quartet1_reg[23]}}, quartet1_reg};
    wire signed [24:0] cached_high_next =
        {{1{quartet2_reg[23]}}, quartet2_reg} + {{1{quartet3_reg[23]}}, quartet3_reg};
    wire signed [25:0] cached_total_next =
        {{1{group_low_reg[23]}}, group_low_reg} + {{1{group_high_reg[23]}}, group_high_reg};
    wire signed [31:0] group_sum_extended = {{7{group_sum_reg[24]}}, group_sum_reg};
    wire signed [31:0] accumulator_next = accumulator + group_sum_extended;

    assign s_axis_tready = (state == ST_BATCH_HEADER)
                         || (state == ST_MATRIX_HEADER)
                         || (state == ST_LOAD_X)
                         || ((state == ST_LOAD_Q) && (active_mode != MODE_CACHE_RUN));

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_BATCH_HEADER;
            batch_header_index <= 1'b0;
            batch_count_lo <= 8'd0;
            batch_count <= 16'd0;
            operation_index <= 16'd0;
            header_index <= 2'd0;
            header_k_lo <= 8'd0;
            header_k_hi <= 8'd0;
            header_rows_lo <= 8'd0;
            active_k <= 16'd0;
            active_rows <= 16'd0;
            active_mode <= MODE_INLINE;
            cache_slot <= 5'd0;
            cache_address <= 18'd0;
            cache_valid <= {CACHE_SLOTS{1'b0}};
            activation_stage0 <= 64'd0;
            activation_stage1 <= 64'd0;
            activation_stage2 <= 64'd0;
            activation_stage3 <= 64'd0;
            activation_count <= {K_INDEX_W{1'b0}};
            activation_byte_index <= 1'b0;
            activation_byte_low <= 8'd0;
            row_index <= {ROW_INDEX_W{1'b0}};
            group_index <= {GROUP_INDEX_W{1'b0}};
            last_group_index <= {GROUP_INDEX_W{1'b0}};
            qbyte_index <= 2'd0;
            packed_byte0 <= 8'd0;
            packed_byte1 <= 8'd0;
            q0 <= 6'sd0;
            q1 <= 6'sd0;
            q2 <= 6'sd0;
            q3 <= 6'sd0;
            q4 <= 6'sd0;
            q5 <= 6'sd0;
            q6 <= 6'sd0;
            q7 <= 6'sd0;
            q8 <= 6'sd0;
            q9 <= 6'sd0;
            q10 <= 6'sd0;
            q11 <= 6'sd0;
            q12 <= 6'sd0;
            q13 <= 6'sd0;
            q14 <= 6'sd0;
            q15 <= 6'sd0;
            x0 <= 8'sd0;
            x1 <= 8'sd0;
            x2 <= 8'sd0;
            x3 <= 8'sd0;
            x4 <= 8'sd0;
            x5 <= 8'sd0;
            x6 <= 8'sd0;
            x7 <= 8'sd0;
            x8 <= 16'sd0;
            x9 <= 16'sd0;
            x10 <= 16'sd0;
            x11 <= 16'sd0;
            x12 <= 16'sd0;
            x13 <= 16'sd0;
            x14 <= 16'sd0;
            x15 <= 16'sd0;
            product0_reg <= 22'sd0;
            product1_reg <= 22'sd0;
            product2_reg <= 22'sd0;
            product3_reg <= 22'sd0;
            product4_reg <= 22'sd0;
            product5_reg <= 22'sd0;
            product6_reg <= 22'sd0;
            product7_reg <= 22'sd0;
            product8_reg <= 22'sd0;
            product9_reg <= 22'sd0;
            product10_reg <= 22'sd0;
            product11_reg <= 22'sd0;
            product12_reg <= 22'sd0;
            product13_reg <= 22'sd0;
            product14_reg <= 22'sd0;
            product15_reg <= 22'sd0;
            quartet0_reg <= 24'sd0;
            quartet1_reg <= 24'sd0;
            quartet2_reg <= 24'sd0;
            quartet3_reg <= 24'sd0;
            group_low_reg <= 24'sd0;
            group_high_reg <= 24'sd0;
            group_sum_reg <= 24'sd0;
            accumulator <= 32'sd0;
            m_axis_tdata <= 32'd0;
            m_axis_tvalid <= 1'b0;
            m_axis_tlast <= 1'b0;
            protocol_error <= 1'b0;
        end else begin
            if (input_fire) begin
                if (s_axis_tlast != input_last_expected)
                    protocol_error <= 1'b1;

                case (state)
                    ST_BATCH_HEADER: begin
                        if (!batch_header_index) begin
                            batch_count_lo <= s_axis_tdata;
                            batch_header_index <= 1'b1;
                        end else begin
                            batch_header_index <= 1'b0;
                            if ((packet_batch_count == 0) || (packet_batch_count > MAX_BATCH)) begin
                                protocol_error <= 1'b1;
                                state <= ST_BATCH_HEADER;
                            end else begin
                                batch_count <= packet_batch_count;
                                operation_index <= 16'd0;
                                header_index <= 2'd0;
                                state <= ST_MATRIX_HEADER;
                            end
                        end
                    end

                    ST_MATRIX_HEADER: begin
                        case (header_index)
                            2'd0: begin
                                header_k_lo <= s_axis_tdata;
                                header_index <= 2'd1;
                            end
                            2'd1: begin
                                header_k_hi <= s_axis_tdata;
                                header_index <= 2'd2;
                            end
                            2'd2: begin
                                header_rows_lo <= s_axis_tdata;
                                header_index <= 2'd3;
                            end
                            default: begin
                                header_index <= 2'd0;
                                if (packet_k == 16'd0) begin
                                    if (packet_rows >= CACHE_SLOTS) begin
                                        protocol_error <= 1'b1;
                                        state <= ST_BATCH_HEADER;
                                    end else begin
                                        active_k <= 16'd128;
                                        active_rows <= cache_rows(packet_rows[4:0]);
                                        active_mode <= MODE_CACHE_LOAD;
                                        cache_slot <= packet_rows[4:0];
                                        cache_address <= cache_base_address(packet_rows[4:0]);
                                        activation_count <= {K_INDEX_W{1'b0}};
                                        row_index <= {ROW_INDEX_W{1'b0}};
                                        last_group_index <= (16'd128 >> 2) - 1'b1;
                                        qbyte_index <= 2'd0;
                                        accumulator <= 32'sd0;
                                        state <= ST_LOAD_Q;
                                    end
                                end else if (packet_k == 16'hffff) begin
                                    if ((packet_rows >= CACHE_SLOTS)
                                        || !cache_valid[packet_rows[4:0]]) begin
                                        protocol_error <= 1'b1;
                                        state <= ST_BATCH_HEADER;
                                    end else begin
                                        active_k <= 16'd128;
                                        active_rows <= cache_rows(packet_rows[4:0]);
                                        active_mode <= MODE_CACHE_RUN;
                                        cache_slot <= packet_rows[4:0];
                                        cache_address <= cache_base_address(packet_rows[4:0]);
                                        activation_count <= {K_INDEX_W{1'b0}};
                                        activation_byte_index <= 1'b0;
                                        row_index <= {ROW_INDEX_W{1'b0}};
                                        last_group_index <= (16'd128 >> 4) - 1'b1;
                                        qbyte_index <= 2'd0;
                                        accumulator <= 32'sd0;
                                        state <= ST_LOAD_X;
                                    end
                                end else if ((packet_k < 4) || (packet_k > MAX_K)
                                    || (packet_k[1:0] != 2'b00)
                                    || (packet_rows < 1) || (packet_rows > MAX_ROWS)) begin
                                    protocol_error <= 1'b1;
                                    state <= ST_BATCH_HEADER;
                                end else begin
                                    active_k <= packet_k;
                                    active_rows <= packet_rows;
                                    active_mode <= MODE_INLINE;
                                    activation_count <= {K_INDEX_W{1'b0}};
                                    activation_byte_index <= 1'b0;
                                    row_index <= {ROW_INDEX_W{1'b0}};
                                    last_group_index <= (packet_k >> 2) - 1'b1;
                                    qbyte_index <= 2'd0;
                                    accumulator <= 32'sd0;
                                    state <= ST_LOAD_X;
                                end
                            end
                        endcase
                    end

                    ST_LOAD_X: begin
                        if (!activation_byte_index) begin
                            activation_byte_low <= s_axis_tdata;
                            activation_byte_index <= 1'b1;
                        end else begin
                            if (active_mode == MODE_CACHE_RUN) begin
                                case (activation_count[1:0])
                                    2'd0: activation_stage0[activation_count[3:2] * 16 +: 16] <= $signed({s_axis_tdata, activation_byte_low});
                                    2'd1: activation_stage1[activation_count[3:2] * 16 +: 16] <= $signed({s_axis_tdata, activation_byte_low});
                                    2'd2: activation_stage2[activation_count[3:2] * 16 +: 16] <= $signed({s_axis_tdata, activation_byte_low});
                                    default: activation_stage3[activation_count[3:2] * 16 +: 16] <= $signed({s_axis_tdata, activation_byte_low});
                                endcase
                                if (activation_count[3:0] == 4'd15) begin
                                    activation_word_bank0[activation_count[K_INDEX_W-1:4]] <= activation_stage0;
                                    activation_word_bank1[activation_count[K_INDEX_W-1:4]] <= activation_stage1;
                                    activation_word_bank2[activation_count[K_INDEX_W-1:4]] <= activation_stage2;
                                    activation_word_bank3[activation_count[K_INDEX_W-1:4]] <= {
                                        $signed({s_axis_tdata, activation_byte_low}),
                                        activation_stage3[47:0]
                                    };
                                end
                            end else begin
                                case (activation_count[1:0])
                                    2'd0: activation_bank0[activation_count[K_INDEX_W-1:2]] <= $signed({s_axis_tdata, activation_byte_low});
                                    2'd1: activation_bank1[activation_count[K_INDEX_W-1:2]] <= $signed({s_axis_tdata, activation_byte_low});
                                    2'd2: activation_bank2[activation_count[K_INDEX_W-1:2]] <= $signed({s_axis_tdata, activation_byte_low});
                                    default: activation_bank3[activation_count[K_INDEX_W-1:2]] <= $signed({s_axis_tdata, activation_byte_low});
                                endcase
                            end
                            activation_byte_index <= 1'b0;
                            if (activation_count == active_k - 1'b1) begin
                                activation_count <= {K_INDEX_W{1'b0}};
                                row_index <= {ROW_INDEX_W{1'b0}};
                                group_index <= {GROUP_INDEX_W{1'b0}};
                                qbyte_index <= 2'd0;
                                accumulator <= 32'sd0;
                                state <= (active_mode == MODE_CACHE_RUN) ? ST_CACHE_READ : ST_LOAD_Q;
                            end else begin
                                activation_count <= activation_count + 1'b1;
                            end
                        end
                    end

                    ST_LOAD_Q: begin
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
                                qbyte_index <= 2'd0;
                                if (active_mode == MODE_CACHE_LOAD) begin
                                    cache_address <= cache_address + 1'b1;
                                    if (group_index == (active_k >> 2) - 1'b1) begin
                                        group_index <= {GROUP_INDEX_W{1'b0}};
                                        if (row_index == active_rows - 1'b1) begin
                                            cache_valid[cache_slot] <= 1'b1;
                                            if (operation_index == batch_count - 1'b1) begin
                                                operation_index <= 16'd0;
                                                batch_count <= 16'd0;
                                                batch_header_index <= 1'b0;
                                                state <= ST_BATCH_HEADER;
                                            end else begin
                                                operation_index <= operation_index + 1'b1;
                                                header_index <= 2'd0;
                                                state <= ST_MATRIX_HEADER;
                                            end
                                        end else begin
                                            row_index <= row_index + 1'b1;
                                            state <= ST_LOAD_Q;
                                        end
                                    end else begin
                                        group_index <= group_index + 1'b1;
                                        state <= ST_LOAD_Q;
                                    end
                                end else begin
                                    q0 <= $signed(packed_byte0[5:0]);
                                    q1 <= $signed({packed_byte1[3:0], packed_byte0[7:6]});
                                    q2 <= $signed({s_axis_tdata[1:0], packed_byte1[7:4]});
                                    q3 <= $signed(s_axis_tdata[7:2]);
                                    x0 <= activation_bank0[group_index];
                                    x1 <= activation_bank1[group_index];
                                    x2 <= activation_bank2[group_index];
                                    x3 <= activation_bank3[group_index];
                                    state <= ST_MAC_MULT;
                                end
                            end
                        endcase
                    end

                    default: begin
                        protocol_error <= 1'b1;
                        state <= ST_BATCH_HEADER;
                    end
                endcase
            end

            if (state == ST_CACHE_READ) begin
                state <= ST_CACHE_DECODE;
            end

            if (state == ST_CACHE_DECODE) begin
                q0 <= $signed(cache_read_bank0[5:0]);
                q1 <= $signed(cache_read_bank0[11:6]);
                q2 <= $signed(cache_read_bank0[17:12]);
                q3 <= $signed(cache_read_bank0[23:18]);
                q4 <= $signed(cache_read_bank1[5:0]);
                q5 <= $signed(cache_read_bank1[11:6]);
                q6 <= $signed(cache_read_bank1[17:12]);
                q7 <= $signed(cache_read_bank1[23:18]);
                q8 <= $signed(cache_read_bank2[5:0]);
                q9 <= $signed(cache_read_bank2[11:6]);
                q10 <= $signed(cache_read_bank2[17:12]);
                q11 <= $signed(cache_read_bank2[23:18]);
                q12 <= $signed(cache_read_bank3[5:0]);
                q13 <= $signed(cache_read_bank3[11:6]);
                q14 <= $signed(cache_read_bank3[17:12]);
                q15 <= $signed(cache_read_bank3[23:18]);
                x0 <= activation_read_bank0[15:0];
                x1 <= activation_read_bank1[15:0];
                x2 <= activation_read_bank2[15:0];
                x3 <= activation_read_bank3[15:0];
                x4 <= activation_read_bank0[31:16];
                x5 <= activation_read_bank1[31:16];
                x6 <= activation_read_bank2[31:16];
                x7 <= activation_read_bank3[31:16];
                x8 <= activation_read_bank0[47:32];
                x9 <= activation_read_bank1[47:32];
                x10 <= activation_read_bank2[47:32];
                x11 <= activation_read_bank3[47:32];
                x12 <= activation_read_bank0[63:48];
                x13 <= activation_read_bank1[63:48];
                x14 <= activation_read_bank2[63:48];
                x15 <= activation_read_bank3[63:48];
                state <= ST_MAC_MULT;
            end

            if (state == ST_MAC_MULT) begin
                product0_reg <= product0_next;
                product1_reg <= product1_next;
                product2_reg <= product2_next;
                product3_reg <= product3_next;
                product4_reg <= product4_next;
                product5_reg <= product5_next;
                product6_reg <= product6_next;
                product7_reg <= product7_next;
                product8_reg <= product8_next;
                product9_reg <= product9_next;
                product10_reg <= product10_next;
                product11_reg <= product11_next;
                product12_reg <= product12_next;
                product13_reg <= product13_next;
                product14_reg <= product14_next;
                product15_reg <= product15_next;
                state <= ST_MAC_SUM;
            end

            if (state == ST_MAC_SUM) begin
                if (active_mode == MODE_CACHE_RUN) begin
                    quartet0_reg <= quartet0_next;
                    quartet1_reg <= quartet1_next;
                    quartet2_reg <= quartet2_next;
                    quartet3_reg <= quartet3_next;
                    state <= ST_MAC_MERGE;
                end else begin
                    group_sum_reg <= group_sum_next;
                    state <= ST_MAC_ACC;
                end
            end

            if (state == ST_MAC_MERGE) begin
                if (active_mode == MODE_CACHE_RUN) begin
                    group_low_reg <= cached_low_next[23:0];
                    group_high_reg <= cached_high_next[23:0];
                    state <= ST_MAC_FINAL;
                end else begin
                    group_sum_reg <= group_total_next[24:0];
                    state <= ST_MAC_ACC;
                end
            end

            if (state == ST_MAC_FINAL) begin
                group_sum_reg <= cached_total_next[24:0];
                state <= ST_MAC_ACC;
            end

            if (state == ST_MAC_ACC) begin
                accumulator <= accumulator_next;
                if (active_mode == MODE_CACHE_RUN)
                    cache_address <= cache_address + 3'd4;
                if (group_index == last_group_index) begin
                    m_axis_tdata <= accumulator_next;
                    m_axis_tvalid <= 1'b1;
                    m_axis_tlast <= (operation_index == batch_count - 1'b1)
                                 && (row_index == active_rows - 1'b1);
                    state <= ST_OUTPUT;
                end else begin
                    group_index <= group_index + 1'b1;
                    state <= (active_mode == MODE_CACHE_RUN) ? ST_CACHE_READ : ST_LOAD_Q;
                end
            end

            if ((state == ST_OUTPUT) && m_axis_tready) begin
                m_axis_tvalid <= 1'b0;
                m_axis_tlast <= 1'b0;
                accumulator <= 32'sd0;
                group_index <= {GROUP_INDEX_W{1'b0}};
                qbyte_index <= 2'd0;
                if (row_index == active_rows - 1'b1) begin
                    row_index <= {ROW_INDEX_W{1'b0}};
                    activation_count <= {K_INDEX_W{1'b0}};
                    if (operation_index == batch_count - 1'b1) begin
                        operation_index <= 16'd0;
                        batch_count <= 16'd0;
                        batch_header_index <= 1'b0;
                        state <= ST_BATCH_HEADER;
                    end else begin
                        operation_index <= operation_index + 1'b1;
                        header_index <= 2'd0;
                        state <= ST_MATRIX_HEADER;
                    end
                end else begin
                    row_index <= row_index + 1'b1;
                    state <= (active_mode == MODE_CACHE_RUN) ? ST_CACHE_READ : ST_LOAD_Q;
                end
            end
        end
    end

    always @(posedge clk) begin
        if ((state == ST_LOAD_Q) && input_fire && (qbyte_index == 2'd2)
            && (active_mode == MODE_CACHE_LOAD)) begin
            case (cache_address[1:0])
                2'd0: q6_cache_bank0[cache_address[17:2]] <= {
                    s_axis_tdata, packed_byte1, packed_byte0
                };
                2'd1: q6_cache_bank1[cache_address[17:2]] <= {
                    s_axis_tdata, packed_byte1, packed_byte0
                };
                2'd2: q6_cache_bank2[cache_address[17:2]] <= {
                    s_axis_tdata, packed_byte1, packed_byte0
                };
                default: q6_cache_bank3[cache_address[17:2]] <= {
                    s_axis_tdata, packed_byte1, packed_byte0
                };
            endcase
        end

        if (state == ST_CACHE_READ) begin
            cache_read_bank0 <= q6_cache_bank0[cache_address[17:2]];
            cache_read_bank1 <= q6_cache_bank1[cache_address[17:2]];
            cache_read_bank2 <= q6_cache_bank2[cache_address[17:2]];
            cache_read_bank3 <= q6_cache_bank3[cache_address[17:2]];
            activation_read_bank0 <= activation_word_bank0[group_index];
            activation_read_bank1 <= activation_word_bank1[group_index];
            activation_read_bank2 <= activation_word_bank2[group_index];
            activation_read_bank3 <= activation_word_bank3[group_index];
        end
    end

endmodule

`default_nettype wire
