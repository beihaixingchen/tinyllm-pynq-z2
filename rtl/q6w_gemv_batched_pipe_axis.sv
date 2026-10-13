`timescale 1ns/1ps
`default_nettype none

module q6w_gemv_batched_pipe_axis #(
    parameter integer MAX_K = 352,
    parameter integer MAX_ROWS = 4096,
    parameter integer MAX_BATCH = 32,
    parameter integer CACHE_SLOTS = 10,
    parameter integer CACHE_WORDS = 112640,
    parameter integer CACHE_MATRIX_ROWS = 352,
    parameter integer CACHE_LANES = 16
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
    localparam integer ROW_COUNT_W = (MAX_ROWS <= 1) ? 1 : $clog2(MAX_ROWS + 1);
    localparam integer BATCH_COUNT_W = (MAX_BATCH < 1) ? 1 : $clog2(MAX_BATCH + 1);
    localparam integer OP_INDEX_W = (MAX_BATCH <= 1) ? 1 : $clog2(MAX_BATCH);
    localparam integer GROUP_INDEX_W = (MAX_K <= 4) ? 1 : $clog2(MAX_K / 4);
    localparam [GROUP_INDEX_W-1:0] CACHE_RUN_LAST_GROUP = (128 / CACHE_LANES) - 1;
    localparam integer CACHE_BANK_WORDS = CACHE_WORDS / 4;
    localparam integer CACHE32_BANK_WORDS = CACHE_WORDS / 8;

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
    localparam [3:0] ST_CACHE_PIPE = 4'd12;
    localparam [3:0] ST_MAC_SUM2 = 4'd13;
    localparam [3:0] ST_MAC_MULT_WAIT = 4'd14;
    localparam [3:0] ST_MAC_MULT_WAIT2 = 4'd15;

    localparam [1:0] MODE_INLINE = 2'd0;
    localparam [1:0] MODE_CACHE_LOAD = 2'd1;
    localparam [1:0] MODE_CACHE_RUN = 2'd2;
    localparam [15:0] CACHE_RUN_REUSE_K = 16'hfffe;

    reg [3:0] state;
    reg batch_header_index;
    reg [7:0] batch_count_lo;
    reg [BATCH_COUNT_W-1:0] batch_count;
    reg [OP_INDEX_W-1:0] operation_index;
    reg [OP_INDEX_W-1:0] last_operation_index;
    reg [1:0] header_index;
    reg [7:0] header_k_lo;
    reg [7:0] header_k_hi;
    reg [1:0] header_mode;
    reg [7:0] header_rows_lo;
    reg [15:0] active_k;
    reg [K_INDEX_W-1:0] last_activation_index;
    reg activation_is_last;
    reg [ROW_COUNT_W-1:0] active_rows;
    reg [ROW_COUNT_W-1:0] rows_remaining;
    reg [1:0] active_mode;
    reg [4:0] cache_slot;
    reg [17:0] cache_address;
    reg [CACHE_SLOTS-1:0] cache_valid;
    reg [23:0] cache_read_bank0, cache_read_bank1, cache_read_bank2, cache_read_bank3;
    reg [23:0] cache_read_bank4, cache_read_bank5, cache_read_bank6, cache_read_bank7;
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
    reg [63:0] activation_read_bank0_hi, activation_read_bank1_hi;
    reg [63:0] activation_read_bank2_hi, activation_read_bank3_hi;
    reg [K_INDEX_W-1:0] activation_count;
    reg activation_byte_index;
    reg [7:0] activation_byte_low;
    reg [ROW_INDEX_W-1:0] row_index;
    reg row_is_last;
    reg [GROUP_INDEX_W-1:0] group_index;
    reg [GROUP_INDEX_W-1:0] last_group_index;
    reg [1:0] qbyte_index;
    reg [7:0] packed_byte0;
    reg [7:0] packed_byte1;
    reg input_valid;
    reg [7:0] input_data;
    reg input_last;
    reg cache_write_pending;
    reg [17:0] cache_write_address_reg;
    reg [2:0] cache_write_bank_reg;
    reg [7:0] cache_write_bank_onehot_reg;
    reg [23:0] cache_write_data_reg;
    (* keep = "true" *) reg signed [5:0] q0, q1, q2, q3, q4, q5, q6, q7;
    (* keep = "true" *) reg signed [5:0] q8, q9, q10, q11, q12, q13, q14, q15;
    (* keep = "true" *) reg signed [5:0] q_stage0, q_stage1, q_stage2, q_stage3;
    (* keep = "true" *) reg signed [5:0] q_stage4, q_stage5, q_stage6, q_stage7;
    (* keep = "true" *) reg signed [5:0] q_stage8, q_stage9, q_stage10, q_stage11;
    (* keep = "true" *) reg signed [5:0] q_stage12, q_stage13, q_stage14, q_stage15;
    (* keep = "true" *) reg signed [15:0] x0, x1, x2, x3, x4, x5, x6, x7;
    (* keep = "true" *) reg signed [15:0] x8, x9, x10, x11, x12, x13, x14, x15;
    (* keep = "true" *) reg signed [15:0] x_stage0, x_stage1, x_stage2, x_stage3;
    (* keep = "true" *) reg signed [15:0] x_stage4, x_stage5, x_stage6, x_stage7;
    (* keep = "true" *) reg signed [15:0] x_stage8, x_stage9, x_stage10, x_stage11;
    (* keep = "true" *) reg signed [15:0] x_stage12, x_stage13, x_stage14, x_stage15;
    (* keep = "true" *) reg signed [5:0] q16, q17, q18, q19, q20, q21, q22, q23;
    (* keep = "true" *) reg signed [5:0] q24, q25, q26, q27, q28, q29, q30, q31;
    (* keep = "true" *) reg signed [5:0] q_stage16, q_stage17, q_stage18, q_stage19;
    (* keep = "true" *) reg signed [5:0] q_stage20, q_stage21, q_stage22, q_stage23;
    (* keep = "true" *) reg signed [5:0] q_stage24, q_stage25, q_stage26, q_stage27;
    (* keep = "true" *) reg signed [5:0] q_stage28, q_stage29, q_stage30, q_stage31;
    (* keep = "true" *) reg signed [15:0] x16, x17, x18, x19, x20, x21, x22, x23;
    (* keep = "true" *) reg signed [15:0] x24, x25, x26, x27, x28, x29, x30, x31;
    (* keep = "true" *) reg signed [15:0] x_stage16, x_stage17, x_stage18, x_stage19;
    (* keep = "true" *) reg signed [15:0] x_stage20, x_stage21, x_stage22, x_stage23;
    (* keep = "true" *) reg signed [15:0] x_stage24, x_stage25, x_stage26, x_stage27;
    (* keep = "true" *) reg signed [15:0] x_stage28, x_stage29, x_stage30, x_stage31;
    wire signed [5:0] q_lane [0:31];
    wire signed [15:0] x_lane [0:31];
    wire signed [21:0] product_reg_lane [0:31];
    wire signed [21:0] reduce_product [0:31];
    wire [47:0] multiplier_p [0:31];
    reg signed [21:0] product0_reg, product1_reg, product2_reg, product3_reg;
    reg signed [21:0] product4_reg, product5_reg, product6_reg, product7_reg;
    reg signed [21:0] product8_reg, product9_reg, product10_reg, product11_reg;
    reg signed [21:0] product12_reg, product13_reg, product14_reg, product15_reg;
    reg signed [21:0] product16_reg, product17_reg, product18_reg, product19_reg;
    reg signed [21:0] product20_reg, product21_reg, product22_reg, product23_reg;
    reg signed [21:0] product24_reg, product25_reg, product26_reg, product27_reg;
    reg signed [21:0] product28_reg, product29_reg, product30_reg, product31_reg;
    reg signed [22:0] pair_sum0_reg, pair_sum1_reg, pair_sum2_reg, pair_sum3_reg;
    reg signed [22:0] pair_sum4_reg, pair_sum5_reg, pair_sum6_reg, pair_sum7_reg;
    reg signed [22:0] pair_sum8_reg, pair_sum9_reg, pair_sum10_reg, pair_sum11_reg;
    reg signed [22:0] pair_sum12_reg, pair_sum13_reg, pair_sum14_reg, pair_sum15_reg;
    (* use_dsp = "no" *) reg signed [23:0] quartet0_reg, quartet1_reg, quartet2_reg, quartet3_reg;
    (* use_dsp = "no" *) reg signed [23:0] quartet4_reg, quartet5_reg, quartet6_reg, quartet7_reg;
    reg signed [23:0] group_low_reg, group_high_reg;
    reg signed [24:0] group_sum_reg;
    reg signed [31:0] accumulator;
    reg cache_pipe_issue_done;
    (* keep = "true" *) reg cache_pipe_active;
    reg signed [25:0] cache_pipe_group_sum_reg;
    reg cache_pipe_valid0, cache_pipe_valid1, cache_pipe_valid2, cache_pipe_valid3, cache_pipe_valid4, cache_pipe_valid5;
    reg cache_pipe_last0, cache_pipe_last1, cache_pipe_last2, cache_pipe_last3, cache_pipe_last4, cache_pipe_last5;
    reg cache_pipe32_valid0, cache_pipe32_valid1, cache_pipe32_valid2, cache_pipe32_valid3;
    reg cache_pipe32_valid4, cache_pipe32_valid5, cache_pipe32_valid6, cache_pipe32_valid7, cache_pipe32_valid8;
    reg cache_pipe32_valid9, cache_pipe32_valid10;
    reg cache_pipe32_last0, cache_pipe32_last1, cache_pipe32_last2, cache_pipe32_last3;
    reg cache_pipe32_last4, cache_pipe32_last5, cache_pipe32_last6, cache_pipe32_last7, cache_pipe32_last8;
    reg cache_pipe32_last9, cache_pipe32_last10;
    reg signed [24:0] cache_pipe32_sum8_0_reg, cache_pipe32_sum8_1_reg;
    reg signed [24:0] cache_pipe32_sum8_2_reg, cache_pipe32_sum8_3_reg;
    reg signed [25:0] cache_pipe32_sum16_0_reg, cache_pipe32_sum16_1_reg;
    reg signed [26:0] cache_pipe32_group_sum_reg;

    wire core_s_axis_tready = (state == ST_BATCH_HEADER)
                            || (state == ST_MATRIX_HEADER)
                            || (state == ST_LOAD_X)
                            || ((state == ST_LOAD_Q) && (active_mode != MODE_CACHE_RUN));
    wire input_fire = input_valid && core_s_axis_tready;
    wire [15:0] packet_batch_count = {input_data, batch_count_lo};
    wire [15:0] packet_k = {header_k_hi, header_k_lo};
    wire [15:0] incoming_packet_k = {input_data, header_k_lo};
    wire [15:0] packet_rows = {input_data, header_rows_lo};
    wire input_last_expected =
        ((state == ST_LOAD_Q)
         && (active_mode != MODE_CACHE_RUN)
         && (operation_index == last_operation_index)
         && row_is_last
         && (group_index == last_group_index)
         && (qbyte_index == 2'd2))
        || ((state == ST_LOAD_X)
         && (active_mode == MODE_CACHE_RUN)
         && activation_byte_index
              && activation_is_last
              && (operation_index == last_operation_index))
        || ((state == ST_MATRIX_HEADER)
         && (header_index == 2'd3)
         && (packet_k == CACHE_RUN_REUSE_K)
         && (operation_index == last_operation_index));
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
                    5'd10: cache_base_address = 18'd112640;
                    5'd11: cache_base_address = 18'd123904;
                    5'd12: cache_base_address = 18'd135168;
                    5'd13: cache_base_address = 18'd146432;
                    5'd14: cache_base_address = 18'd157696;
                    default: cache_base_address = 18'd0;
                endcase
            end
        end
    endfunction
    function [15:0] cache_rows;
        input [4:0] slot;
        begin
            if ((CACHE_SLOTS > 10) && (slot >= 10))
                cache_rows = 16'd256;
            else
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
    wire signed [21:0] product16_next = q16 * x16;
    wire signed [21:0] product17_next = q17 * x17;
    wire signed [21:0] product18_next = q18 * x18;
    wire signed [21:0] product19_next = q19 * x19;
    wire signed [21:0] product20_next = q20 * x20;
    wire signed [21:0] product21_next = q21 * x21;
    wire signed [21:0] product22_next = q22 * x22;
    wire signed [21:0] product23_next = q23 * x23;
    wire signed [21:0] product24_next = q24 * x24;
    wire signed [21:0] product25_next = q25 * x25;
    wire signed [21:0] product26_next = q26 * x26;
    wire signed [21:0] product27_next = q27 * x27;
    wire signed [21:0] product28_next = q28 * x28;
    wire signed [21:0] product29_next = q29 * x29;
    wire signed [21:0] product30_next = q30 * x30;
    wire signed [21:0] product31_next = q31 * x31;
    assign q_lane[0] = q0; assign q_lane[1] = q1; assign q_lane[2] = q2; assign q_lane[3] = q3;
    assign q_lane[4] = q4; assign q_lane[5] = q5; assign q_lane[6] = q6; assign q_lane[7] = q7;
    assign q_lane[8] = q8; assign q_lane[9] = q9; assign q_lane[10] = q10; assign q_lane[11] = q11;
    assign q_lane[12] = q12; assign q_lane[13] = q13; assign q_lane[14] = q14; assign q_lane[15] = q15;
    assign q_lane[16] = q16; assign q_lane[17] = q17; assign q_lane[18] = q18; assign q_lane[19] = q19;
    assign q_lane[20] = q20; assign q_lane[21] = q21; assign q_lane[22] = q22; assign q_lane[23] = q23;
    assign q_lane[24] = q24; assign q_lane[25] = q25; assign q_lane[26] = q26; assign q_lane[27] = q27;
    assign q_lane[28] = q28; assign q_lane[29] = q29; assign q_lane[30] = q30; assign q_lane[31] = q31;
    assign x_lane[0] = x0; assign x_lane[1] = x1; assign x_lane[2] = x2; assign x_lane[3] = x3;
    assign x_lane[4] = x4; assign x_lane[5] = x5; assign x_lane[6] = x6; assign x_lane[7] = x7;
    assign x_lane[8] = x8; assign x_lane[9] = x9; assign x_lane[10] = x10; assign x_lane[11] = x11;
    assign x_lane[12] = x12; assign x_lane[13] = x13; assign x_lane[14] = x14; assign x_lane[15] = x15;
    assign x_lane[16] = x16; assign x_lane[17] = x17; assign x_lane[18] = x18; assign x_lane[19] = x19;
    assign x_lane[20] = x20; assign x_lane[21] = x21; assign x_lane[22] = x22; assign x_lane[23] = x23;
    assign x_lane[24] = x24; assign x_lane[25] = x25; assign x_lane[26] = x26; assign x_lane[27] = x27;
    assign x_lane[28] = x28; assign x_lane[29] = x29; assign x_lane[30] = x30; assign x_lane[31] = x31;
    assign product_reg_lane[0] = product0_reg; assign product_reg_lane[1] = product1_reg;
    assign product_reg_lane[2] = product2_reg; assign product_reg_lane[3] = product3_reg;
    assign product_reg_lane[4] = product4_reg; assign product_reg_lane[5] = product5_reg;
    assign product_reg_lane[6] = product6_reg; assign product_reg_lane[7] = product7_reg;
    assign product_reg_lane[8] = product8_reg; assign product_reg_lane[9] = product9_reg;
    assign product_reg_lane[10] = product10_reg; assign product_reg_lane[11] = product11_reg;
    assign product_reg_lane[12] = product12_reg; assign product_reg_lane[13] = product13_reg;
    assign product_reg_lane[14] = product14_reg; assign product_reg_lane[15] = product15_reg;
    assign product_reg_lane[16] = product16_reg; assign product_reg_lane[17] = product17_reg;
    assign product_reg_lane[18] = product18_reg; assign product_reg_lane[19] = product19_reg;
    assign product_reg_lane[20] = product20_reg; assign product_reg_lane[21] = product21_reg;
    assign product_reg_lane[22] = product22_reg; assign product_reg_lane[23] = product23_reg;
    assign product_reg_lane[24] = product24_reg; assign product_reg_lane[25] = product25_reg;
    assign product_reg_lane[26] = product26_reg; assign product_reg_lane[27] = product27_reg;
    assign product_reg_lane[28] = product28_reg; assign product_reg_lane[29] = product29_reg;
    assign product_reg_lane[30] = product30_reg; assign product_reg_lane[31] = product31_reg;

    generate
        if (CACHE_LANES == 32) begin : gen_dsp48_lanes
            genvar lane;
            for (lane = 0; lane < 32; lane = lane + 1) begin : gen_dsp48_lane
                DSP48E1 #(
                    .ACASCREG(1),
                    .AREG(1),
                    .BCASCREG(1),
                    .BREG(1),
                    .MREG(1),
                    .ALUMODEREG(0),
                    .CARRYINREG(0),
                    .CARRYINSELREG(0),
                    .INMODEREG(0),
                    .OPMODEREG(0),
                    .PREG(1),
                    .USE_MULT("MULTIPLY"),
                    .USE_SIMD("ONE48")
                ) multiplier (
                    .A({{24{q_lane[lane][5]}}, q_lane[lane]}),
                    .ACIN(30'd0),
                    .ALUMODE(4'b0000),
                    .B({{2{x_lane[lane][15]}}, x_lane[lane]}),
                    .BCIN(18'd0),
                    .C(48'd0),
                    .CARRYCASCIN(1'b0),
                    .CARRYIN(1'b0),
                    .CARRYINSEL(3'b000),
                    .CEA1(1'b1),
                    .CEA2(1'b1),
                    .CEAD(1'b0),
                    .CEALUMODE(1'b0),
                    .CEB1(1'b1),
                    .CEB2(1'b1),
                    .CEC(1'b0),
                    .CECTRL(1'b0),
                    .CED(1'b0),
                    .CEINMODE(1'b0),
                    .CEM(1'b1),
                    .CEP(1'b1),
                    .CLK(clk),
                    .D(25'd0),
                    .INMODE(5'b00000),
                    .MULTSIGNIN(1'b0),
                    .OPMODE(7'b0000101),
                    .PCIN(48'd0),
                    .RSTA(1'b0),
                    .RSTALLCARRYIN(1'b0),
                    .RSTALUMODE(1'b0),
                    .RSTB(1'b0),
                    .RSTC(1'b0),
                    .RSTCTRL(1'b0),
                    .RSTD(1'b0),
                    .RSTINMODE(1'b0),
                    .RSTM(1'b0),
                    .RSTP(1'b0),
                    .P(multiplier_p[lane])
                );
                assign reduce_product[lane] = $signed(multiplier_p[lane][21:0]);
            end
        end else begin : gen_inferred_multiplier_lanes
            genvar lane;
            for (lane = 0; lane < 32; lane = lane + 1) begin : gen_inferred_multiplier_lane
                assign multiplier_p[lane] = 48'd0;
                assign reduce_product[lane] = product_reg_lane[lane];
            end
        end
    endgenerate
    wire signed [22:0] pair_sum0_next =
        {{1{reduce_product[0][21]}}, reduce_product[0]} + {{1{reduce_product[1][21]}}, reduce_product[1]};
    wire signed [22:0] pair_sum1_next =
        {{1{reduce_product[2][21]}}, reduce_product[2]} + {{1{reduce_product[3][21]}}, reduce_product[3]};
    wire signed [23:0] group_sum_pair_next =
        {{1{pair_sum0_reg[22]}}, pair_sum0_reg} + {{1{pair_sum1_reg[22]}}, pair_sum1_reg};
    wire signed [22:0] pair_sum2_next =
        {{1{reduce_product[4][21]}}, reduce_product[4]} + {{1{reduce_product[5][21]}}, reduce_product[5]};
    wire signed [22:0] pair_sum3_next =
        {{1{reduce_product[6][21]}}, reduce_product[6]} + {{1{reduce_product[7][21]}}, reduce_product[7]};
    wire signed [23:0] group_high_next =
        {{1{pair_sum2_next[22]}}, pair_sum2_next} + {{1{pair_sum3_next[22]}}, pair_sum3_next};
    wire signed [24:0] group_total_next =
        {{1{group_low_reg[23]}}, group_low_reg} + {{1{group_high_reg[23]}}, group_high_reg};
    wire signed [22:0] pair_sum4_next =
        {{1{reduce_product[8][21]}}, reduce_product[8]} + {{1{reduce_product[9][21]}}, reduce_product[9]};
    wire signed [22:0] pair_sum5_next =
        {{1{reduce_product[10][21]}}, reduce_product[10]} + {{1{reduce_product[11][21]}}, reduce_product[11]};
    wire signed [22:0] pair_sum6_next =
        {{1{reduce_product[12][21]}}, reduce_product[12]} + {{1{reduce_product[13][21]}}, reduce_product[13]};
    wire signed [22:0] pair_sum7_next =
        {{1{reduce_product[14][21]}}, reduce_product[14]} + {{1{reduce_product[15][21]}}, reduce_product[15]};
    (* use_dsp = "no" *) wire signed [23:0] quartet0_next =
        {{1{pair_sum0_next[22]}}, pair_sum0_next} + {{1{pair_sum1_next[22]}}, pair_sum1_next};
    (* use_dsp = "no" *) wire signed [23:0] quartet1_next =
        {{1{pair_sum2_next[22]}}, pair_sum2_next} + {{1{pair_sum3_next[22]}}, pair_sum3_next};
    (* use_dsp = "no" *) wire signed [23:0] quartet2_next =
        {{1{pair_sum4_next[22]}}, pair_sum4_next} + {{1{pair_sum5_next[22]}}, pair_sum5_next};
    (* use_dsp = "no" *) wire signed [23:0] quartet3_next =
        {{1{pair_sum6_next[22]}}, pair_sum6_next} + {{1{pair_sum7_next[22]}}, pair_sum7_next};
    wire signed [22:0] pair_sum8_next =
        {{1{reduce_product[16][21]}}, reduce_product[16]} + {{1{reduce_product[17][21]}}, reduce_product[17]};
    wire signed [22:0] pair_sum9_next =
        {{1{reduce_product[18][21]}}, reduce_product[18]} + {{1{reduce_product[19][21]}}, reduce_product[19]};
    wire signed [22:0] pair_sum10_next =
        {{1{reduce_product[20][21]}}, reduce_product[20]} + {{1{reduce_product[21][21]}}, reduce_product[21]};
    wire signed [22:0] pair_sum11_next =
        {{1{reduce_product[22][21]}}, reduce_product[22]} + {{1{reduce_product[23][21]}}, reduce_product[23]};
    wire signed [22:0] pair_sum12_next =
        {{1{reduce_product[24][21]}}, reduce_product[24]} + {{1{reduce_product[25][21]}}, reduce_product[25]};
    wire signed [22:0] pair_sum13_next =
        {{1{reduce_product[26][21]}}, reduce_product[26]} + {{1{reduce_product[27][21]}}, reduce_product[27]};
    wire signed [22:0] pair_sum14_next =
        {{1{reduce_product[28][21]}}, reduce_product[28]} + {{1{reduce_product[29][21]}}, reduce_product[29]};
    wire signed [22:0] pair_sum15_next =
        {{1{reduce_product[30][21]}}, reduce_product[30]} + {{1{reduce_product[31][21]}}, reduce_product[31]};
    (* use_dsp = "no" *) wire signed [23:0] quartet4_next =
        {{1{pair_sum8_next[22]}}, pair_sum8_next} + {{1{pair_sum9_next[22]}}, pair_sum9_next};
    (* use_dsp = "no" *) wire signed [23:0] quartet5_next =
        {{1{pair_sum10_next[22]}}, pair_sum10_next} + {{1{pair_sum11_next[22]}}, pair_sum11_next};
    (* use_dsp = "no" *) wire signed [23:0] quartet6_next =
        {{1{pair_sum12_next[22]}}, pair_sum12_next} + {{1{pair_sum13_next[22]}}, pair_sum13_next};
    (* use_dsp = "no" *) wire signed [23:0] quartet7_next =
        {{1{pair_sum14_next[22]}}, pair_sum14_next} + {{1{pair_sum15_next[22]}}, pair_sum15_next};
    (* use_dsp = "no" *) wire signed [23:0] quartet32_0_next =
        {{1{pair_sum0_reg[22]}}, pair_sum0_reg} + {{1{pair_sum1_reg[22]}}, pair_sum1_reg};
    (* use_dsp = "no" *) wire signed [23:0] quartet32_1_next =
        {{1{pair_sum2_reg[22]}}, pair_sum2_reg} + {{1{pair_sum3_reg[22]}}, pair_sum3_reg};
    (* use_dsp = "no" *) wire signed [23:0] quartet32_2_next =
        {{1{pair_sum4_reg[22]}}, pair_sum4_reg} + {{1{pair_sum5_reg[22]}}, pair_sum5_reg};
    (* use_dsp = "no" *) wire signed [23:0] quartet32_3_next =
        {{1{pair_sum6_reg[22]}}, pair_sum6_reg} + {{1{pair_sum7_reg[22]}}, pair_sum7_reg};
    (* use_dsp = "no" *) wire signed [23:0] quartet32_4_next =
        {{1{pair_sum8_reg[22]}}, pair_sum8_reg} + {{1{pair_sum9_reg[22]}}, pair_sum9_reg};
    (* use_dsp = "no" *) wire signed [23:0] quartet32_5_next =
        {{1{pair_sum10_reg[22]}}, pair_sum10_reg} + {{1{pair_sum11_reg[22]}}, pair_sum11_reg};
    (* use_dsp = "no" *) wire signed [23:0] quartet32_6_next =
        {{1{pair_sum12_reg[22]}}, pair_sum12_reg} + {{1{pair_sum13_reg[22]}}, pair_sum13_reg};
    (* use_dsp = "no" *) wire signed [23:0] quartet32_7_next =
        {{1{pair_sum14_reg[22]}}, pair_sum14_reg} + {{1{pair_sum15_reg[22]}}, pair_sum15_reg};
    wire signed [24:0] cached_low_next =
        {{1{quartet0_reg[23]}}, quartet0_reg} + {{1{quartet1_reg[23]}}, quartet1_reg};
    wire signed [24:0] cached_high_next =
        {{1{quartet2_reg[23]}}, quartet2_reg} + {{1{quartet3_reg[23]}}, quartet3_reg};
    wire signed [25:0] cached_total_next =
        {{1{group_low_reg[23]}}, group_low_reg} + {{1{group_high_reg[23]}}, group_high_reg};
    wire signed [31:0] group_sum_extended = {{7{group_sum_reg[24]}}, group_sum_reg};
    wire signed [31:0] accumulator_next = accumulator + group_sum_extended;
    wire signed [25:0] cache_pipe_group_sum =
        {{1{cached_low_next[24]}}, cached_low_next}
        + {{1{cached_high_next[24]}}, cached_high_next};
    wire signed [31:0] cache_pipe_accumulator_next = accumulator
        + {{6{cache_pipe_group_sum_reg[25]}}, cache_pipe_group_sum_reg};
    wire signed [24:0] cache_pipe32_sum8_0_next =
        {{1{quartet0_reg[23]}}, quartet0_reg} + {{1{quartet1_reg[23]}}, quartet1_reg};
    wire signed [24:0] cache_pipe32_sum8_1_next =
        {{1{quartet2_reg[23]}}, quartet2_reg} + {{1{quartet3_reg[23]}}, quartet3_reg};
    wire signed [24:0] cache_pipe32_sum8_2_next =
        {{1{quartet4_reg[23]}}, quartet4_reg} + {{1{quartet5_reg[23]}}, quartet5_reg};
    wire signed [24:0] cache_pipe32_sum8_3_next =
        {{1{quartet6_reg[23]}}, quartet6_reg} + {{1{quartet7_reg[23]}}, quartet7_reg};
    wire signed [25:0] cache_pipe32_sum16_0_next =
        {{1{cache_pipe32_sum8_0_reg[24]}}, cache_pipe32_sum8_0_reg}
        + {{1{cache_pipe32_sum8_1_reg[24]}}, cache_pipe32_sum8_1_reg};
    wire signed [25:0] cache_pipe32_sum16_1_next =
        {{1{cache_pipe32_sum8_2_reg[24]}}, cache_pipe32_sum8_2_reg}
        + {{1{cache_pipe32_sum8_3_reg[24]}}, cache_pipe32_sum8_3_reg};
    wire signed [26:0] cache_pipe32_group_sum_next =
        {{1{cache_pipe32_sum16_0_reg[25]}}, cache_pipe32_sum16_0_reg}
        + {{1{cache_pipe32_sum16_1_reg[25]}}, cache_pipe32_sum16_1_reg};
    wire signed [31:0] cache_pipe32_accumulator_next = accumulator
        + {{5{cache_pipe32_group_sum_reg[26]}}, cache_pipe32_group_sum_reg};
    wire cache_pipe_issue = cache_pipe_active && !cache_pipe_issue_done;
    wire cache_write_capture = (state == ST_LOAD_Q)
        && input_fire
        && (qbyte_index == 2'd2)
        && (active_mode == MODE_CACHE_LOAD);
    wire cache_write_select_capture = (state == ST_LOAD_Q)
        && input_fire
        && (qbyte_index == 2'd0)
        && (active_mode == MODE_CACHE_LOAD);
    wire activation_word_write = (state == ST_LOAD_X)
        && input_fire
        && activation_byte_index
        && (active_mode == MODE_CACHE_RUN)
        && (activation_count[3:0] == 4'd15);
    wire [K_INDEX_W-1:0] activation_word_write_address = activation_count >> 4;
    wire [63:0] activation_word_bank3_write_data = {
        $signed({input_data, activation_byte_low}), activation_stage3[47:0]
    };
    wire [K_INDEX_W-1:0] activation_word_read_address =
        (CACHE_LANES == 32) ? (group_index << 1) : group_index;
    wire [K_INDEX_W-1:0] activation_word_read_address_hi = (group_index << 1) + 1'b1;

    assign s_axis_tready = rst_n && (!input_valid || core_s_axis_tready);

    always @(posedge clk) begin
        if (!rst_n) begin
            input_valid <= 1'b0;
        end else if (s_axis_tready) begin
            input_valid <= s_axis_tvalid;
            if (s_axis_tvalid) begin
                input_data <= s_axis_tdata;
                input_last <= s_axis_tlast;
            end
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            cache_write_pending <= 1'b0;
            cache_write_bank_onehot_reg <= 8'd0;
        end else begin
            cache_write_pending <= cache_write_capture;
            if (cache_write_select_capture)
                cache_write_bank_onehot_reg <= 8'b00000001 << cache_address[2:0];
            if (cache_write_capture) begin
                if (CACHE_LANES == 32) begin
                    cache_write_address_reg <= cache_address[17:3];
                    cache_write_bank_reg <= cache_address[2:0];
                end else begin
                    cache_write_address_reg <= cache_address[17:2];
                    cache_write_bank_reg <= {1'b0, cache_address[1:0]};
                end
                cache_write_data_reg <= {input_data, packed_byte1, packed_byte0};
            end
        end
    end

    generate
        if (CACHE_LANES == 32) begin : gen_cache_banks32
            (* ram_style = "block" *) reg [23:0] q6_cache_bank0 [0:CACHE32_BANK_WORDS-1];
            (* ram_style = "block" *) reg [23:0] q6_cache_bank1 [0:CACHE32_BANK_WORDS-1];
            (* ram_style = "block" *) reg [23:0] q6_cache_bank2 [0:CACHE32_BANK_WORDS-1];
            (* ram_style = "block" *) reg [23:0] q6_cache_bank3 [0:CACHE32_BANK_WORDS-1];
            (* ram_style = "block" *) reg [23:0] q6_cache_bank4 [0:CACHE32_BANK_WORDS-1];
            (* ram_style = "block" *) reg [23:0] q6_cache_bank5 [0:CACHE32_BANK_WORDS-1];
            (* ram_style = "block" *) reg [23:0] q6_cache_bank6 [0:CACHE32_BANK_WORDS-1];
            (* ram_style = "block" *) reg [23:0] q6_cache_bank7 [0:CACHE32_BANK_WORDS-1];

            always @(posedge clk) begin
                if (rst_n && cache_write_pending && cache_write_bank_onehot_reg[0])
                    q6_cache_bank0[cache_write_address_reg] <= cache_write_data_reg;
                else
                    cache_read_bank0 <= q6_cache_bank0[cache_address[17:3]];
            end
            always @(posedge clk) begin
                if (rst_n && cache_write_pending && cache_write_bank_onehot_reg[1])
                    q6_cache_bank1[cache_write_address_reg] <= cache_write_data_reg;
                else
                    cache_read_bank1 <= q6_cache_bank1[cache_address[17:3]];
            end
            always @(posedge clk) begin
                if (rst_n && cache_write_pending && cache_write_bank_onehot_reg[2])
                    q6_cache_bank2[cache_write_address_reg] <= cache_write_data_reg;
                else
                    cache_read_bank2 <= q6_cache_bank2[cache_address[17:3]];
            end
            always @(posedge clk) begin
                if (rst_n && cache_write_pending && cache_write_bank_onehot_reg[3])
                    q6_cache_bank3[cache_write_address_reg] <= cache_write_data_reg;
                else
                    cache_read_bank3 <= q6_cache_bank3[cache_address[17:3]];
            end
            always @(posedge clk) begin
                if (rst_n && cache_write_pending && cache_write_bank_onehot_reg[4])
                    q6_cache_bank4[cache_write_address_reg] <= cache_write_data_reg;
                else
                    cache_read_bank4 <= q6_cache_bank4[cache_address[17:3]];
            end
            always @(posedge clk) begin
                if (rst_n && cache_write_pending && cache_write_bank_onehot_reg[5])
                    q6_cache_bank5[cache_write_address_reg] <= cache_write_data_reg;
                else
                    cache_read_bank5 <= q6_cache_bank5[cache_address[17:3]];
            end
            always @(posedge clk) begin
                if (rst_n && cache_write_pending && cache_write_bank_onehot_reg[6])
                    q6_cache_bank6[cache_write_address_reg] <= cache_write_data_reg;
                else
                    cache_read_bank6 <= q6_cache_bank6[cache_address[17:3]];
            end
            always @(posedge clk) begin
                if (rst_n && cache_write_pending && cache_write_bank_onehot_reg[7])
                    q6_cache_bank7[cache_write_address_reg] <= cache_write_data_reg;
                else
                    cache_read_bank7 <= q6_cache_bank7[cache_address[17:3]];
            end
        end else begin : gen_cache_banks16
            (* ram_style = "block" *) reg [23:0] q6_cache_bank0 [0:CACHE_BANK_WORDS-1];
            (* ram_style = "block" *) reg [23:0] q6_cache_bank1 [0:CACHE_BANK_WORDS-1];
            (* ram_style = "block" *) reg [23:0] q6_cache_bank2 [0:CACHE_BANK_WORDS-1];
            (* ram_style = "block" *) reg [23:0] q6_cache_bank3 [0:CACHE_BANK_WORDS-1];

            always @(posedge clk) begin
                if (rst_n && cache_write_pending && (cache_write_bank_reg == 3'd0))
                    q6_cache_bank0[cache_write_address_reg] <= cache_write_data_reg;
                else
                    cache_read_bank0 <= q6_cache_bank0[cache_address[17:2]];
            end
            always @(posedge clk) begin
                if (rst_n && cache_write_pending && (cache_write_bank_reg == 3'd1))
                    q6_cache_bank1[cache_write_address_reg] <= cache_write_data_reg;
                else
                    cache_read_bank1 <= q6_cache_bank1[cache_address[17:2]];
            end
            always @(posedge clk) begin
                if (rst_n && cache_write_pending && (cache_write_bank_reg == 3'd2))
                    q6_cache_bank2[cache_write_address_reg] <= cache_write_data_reg;
                else
                    cache_read_bank2 <= q6_cache_bank2[cache_address[17:2]];
            end
            always @(posedge clk) begin
                if (rst_n && cache_write_pending && (cache_write_bank_reg == 3'd3))
                    q6_cache_bank3[cache_write_address_reg] <= cache_write_data_reg;
                else
                    cache_read_bank3 <= q6_cache_bank3[cache_address[17:2]];
            end
        end
    endgenerate

    always @(posedge clk) begin
        if (rst_n && activation_word_write)
            activation_word_bank0[activation_word_write_address] <= activation_stage0;
        else
            activation_read_bank0 <= activation_word_bank0[activation_word_read_address];
    end
    always @(posedge clk)
        activation_read_bank0_hi <= activation_word_bank0[activation_word_read_address_hi];

    always @(posedge clk) begin
        if (rst_n && activation_word_write)
            activation_word_bank1[activation_word_write_address] <= activation_stage1;
        else
            activation_read_bank1 <= activation_word_bank1[activation_word_read_address];
    end
    always @(posedge clk)
        activation_read_bank1_hi <= activation_word_bank1[activation_word_read_address_hi];

    always @(posedge clk) begin
        if (rst_n && activation_word_write)
            activation_word_bank2[activation_word_write_address] <= activation_stage2;
        else
            activation_read_bank2 <= activation_word_bank2[activation_word_read_address];
    end
    always @(posedge clk)
        activation_read_bank2_hi <= activation_word_bank2[activation_word_read_address_hi];

    always @(posedge clk) begin
        if (rst_n && activation_word_write)
            activation_word_bank3[activation_word_write_address] <= activation_word_bank3_write_data;
        else
            activation_read_bank3 <= activation_word_bank3[activation_word_read_address];
    end
    always @(posedge clk)
        activation_read_bank3_hi <= activation_word_bank3[activation_word_read_address_hi];

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_BATCH_HEADER;
            batch_header_index <= 1'b0;
            batch_count_lo <= 8'd0;
            batch_count <= {BATCH_COUNT_W{1'b0}};
            last_operation_index <= {OP_INDEX_W{1'b0}};
            operation_index <= 16'd0;
            header_index <= 2'd0;
            header_k_lo <= 8'd0;
            header_k_hi <= 8'd0;
            header_mode <= MODE_INLINE;
            header_rows_lo <= 8'd0;
            active_k <= 16'd0;
            last_activation_index <= {K_INDEX_W{1'b0}};
            activation_is_last <= 1'b0;
            active_rows <= 16'd0;
            rows_remaining <= 16'd0;
            row_is_last <= 1'b0;
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
            accumulator <= 32'sd0;
            cache_pipe_issue_done <= 1'b0;
            cache_pipe_active <= 1'b0;
            cache_pipe_valid0 <= 1'b0;
            cache_pipe_valid1 <= 1'b0;
            cache_pipe_valid2 <= 1'b0;
            cache_pipe_valid3 <= 1'b0;
            cache_pipe_valid4 <= 1'b0;
            cache_pipe_valid5 <= 1'b0;
            cache_pipe_last0 <= 1'b0;
            cache_pipe_last1 <= 1'b0;
            cache_pipe_last2 <= 1'b0;
            cache_pipe_last3 <= 1'b0;
            cache_pipe_last4 <= 1'b0;
            cache_pipe_last5 <= 1'b0;
            cache_pipe32_valid0 <= 1'b0;
            cache_pipe32_valid1 <= 1'b0;
            cache_pipe32_valid2 <= 1'b0;
            cache_pipe32_valid3 <= 1'b0;
            cache_pipe32_valid4 <= 1'b0;
            cache_pipe32_valid5 <= 1'b0;
            cache_pipe32_valid6 <= 1'b0;
            cache_pipe32_valid7 <= 1'b0;
            cache_pipe32_valid8 <= 1'b0;
            cache_pipe32_valid9 <= 1'b0;
            cache_pipe32_valid10 <= 1'b0;
            cache_pipe32_last0 <= 1'b0;
            cache_pipe32_last1 <= 1'b0;
            cache_pipe32_last2 <= 1'b0;
            cache_pipe32_last3 <= 1'b0;
            cache_pipe32_last4 <= 1'b0;
            cache_pipe32_last5 <= 1'b0;
            cache_pipe32_last6 <= 1'b0;
            cache_pipe32_last7 <= 1'b0;
            cache_pipe32_last8 <= 1'b0;
            cache_pipe32_last9 <= 1'b0;
            cache_pipe32_last10 <= 1'b0;
            m_axis_tdata <= 32'd0;
            m_axis_tvalid <= 1'b0;
            m_axis_tlast <= 1'b0;
            protocol_error <= 1'b0;
        end else begin
            if (input_fire) begin
                if (input_last != input_last_expected)
                    protocol_error <= 1'b1;

                case (state)
                    ST_BATCH_HEADER: begin
                        if (!batch_header_index) begin
                            batch_count_lo <= input_data;
                            batch_header_index <= 1'b1;
                        end else begin
                            batch_header_index <= 1'b0;
                            if ((packet_batch_count == 0) || (packet_batch_count > MAX_BATCH)) begin
                                protocol_error <= 1'b1;
                                state <= ST_BATCH_HEADER;
                            end else begin
                                batch_count <= packet_batch_count[BATCH_COUNT_W-1:0];
                                last_operation_index <= packet_batch_count[OP_INDEX_W-1:0] - 1'b1;
                                operation_index <= {OP_INDEX_W{1'b0}};
                                header_index <= 2'd0;
                                state <= ST_MATRIX_HEADER;
                            end
                        end
                    end

                    ST_MATRIX_HEADER: begin
                        case (header_index)
                            2'd0: begin
                                header_k_lo <= input_data;
                                header_index <= 2'd1;
                            end
                            2'd1: begin
                                header_k_hi <= input_data;
                                if ((header_k_lo == 8'd0) && (input_data == 8'd0)) begin
                                    header_mode <= MODE_CACHE_LOAD;
                                    last_activation_index <= 127;
                                end else if ((header_k_lo == 8'hff) && (input_data == 8'hff)) begin
                                    header_mode <= MODE_CACHE_RUN;
                                    last_activation_index <= 127;
                                end else if ((header_k_lo == 8'hfe) && (input_data == 8'hff)) begin
                                    header_mode <= MODE_CACHE_RUN;
                                    last_activation_index <= 127;
                                end else begin
                                    header_mode <= MODE_INLINE;
                                    last_activation_index <= incoming_packet_k[K_INDEX_W-1:0] - 1'b1;
                                end
                                header_index <= 2'd2;
                            end
                            2'd2: begin
                                header_rows_lo <= input_data;
                                row_index <= {ROW_INDEX_W{1'b0}};
                                accumulator <= 32'sd0;
                                active_mode <= header_mode;
                                cache_slot <= input_data[4:0];
                                cache_address <= cache_base_address(input_data[4:0]);
                                if (header_mode == MODE_CACHE_LOAD)
                                    last_group_index <= (16'd128 >> 2) - 1'b1;
                                else if (header_mode == MODE_CACHE_RUN)
                                    last_group_index <= (16'd128 / CACHE_LANES) - 1'b1;
                                else
                                    last_group_index <= (packet_k >> 2) - 1'b1;
                                activation_count <= {K_INDEX_W{1'b0}};
                                activation_is_last <= 1'b0;
                                 if ((header_mode == MODE_CACHE_LOAD)
                                     || (header_mode == MODE_CACHE_RUN)) begin
                                    rows_remaining <= cache_rows(input_data[4:0]);
                                    row_is_last <= (cache_rows(input_data[4:0]) == 16'd1);
                                 end
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
                                        qbyte_index <= 2'd0;
                                        state <= ST_LOAD_Q;
                                    end
                                end else if ((packet_k == 16'hffff)
                                    || (packet_k == CACHE_RUN_REUSE_K)) begin
                                    if ((packet_rows >= CACHE_SLOTS)
                                        || !cache_valid[packet_rows[4:0]]) begin
                                        protocol_error <= 1'b1;
                                        state <= ST_BATCH_HEADER;
                                    end else begin
                                        active_k <= 16'd128;
                                        activation_byte_index <= 1'b0;
                                        qbyte_index <= 2'd0;
                                        cache_pipe_issue_done <= 1'b0;
                                        cache_pipe32_valid0 <= 1'b0;
                                        cache_pipe32_valid1 <= 1'b0;
                                        cache_pipe32_valid2 <= 1'b0;
                                        cache_pipe32_valid3 <= 1'b0;
                                        cache_pipe32_valid4 <= 1'b0;
                                        cache_pipe32_valid5 <= 1'b0;
                                        cache_pipe32_valid6 <= 1'b0;
                                        cache_pipe32_valid7 <= 1'b0;
                                        cache_pipe32_valid8 <= 1'b0;
                                        cache_pipe32_valid9 <= 1'b0;
                                        cache_pipe32_valid10 <= 1'b0;
                                        cache_pipe_valid0 <= 1'b0;
                                        cache_pipe_valid1 <= 1'b0;
                                        cache_pipe_valid2 <= 1'b0;
                                        cache_pipe_valid3 <= 1'b0;
                                        cache_pipe_valid4 <= 1'b0;
                                        cache_pipe_valid5 <= 1'b0;
                                        cache_pipe_last0 <= 1'b0;
                                        cache_pipe_last1 <= 1'b0;
                                        cache_pipe_last2 <= 1'b0;
                                        cache_pipe_last3 <= 1'b0;
                                        cache_pipe_last4 <= 1'b0;
                                        cache_pipe_last5 <= 1'b0;
                                        if (packet_k == CACHE_RUN_REUSE_K) begin
                                            group_index <= {GROUP_INDEX_W{1'b0}};
                                            cache_pipe_issue_done <= 1'b0;
                                            cache_pipe32_valid0 <= 1'b0;
                                            cache_pipe32_valid1 <= 1'b0;
                                            cache_pipe32_valid2 <= 1'b0;
                                            cache_pipe32_valid3 <= 1'b0;
                                            cache_pipe32_valid4 <= 1'b0;
                                            cache_pipe32_valid5 <= 1'b0;
                                            cache_pipe32_valid6 <= 1'b0;
                                            cache_pipe32_valid7 <= 1'b0;
                                            cache_pipe32_valid8 <= 1'b0;
                                            cache_pipe32_valid9 <= 1'b0;
                                            cache_pipe32_valid10 <= 1'b0;
                                            cache_pipe_valid0 <= 1'b0;
                                            cache_pipe_valid1 <= 1'b0;
                                            cache_pipe_valid2 <= 1'b0;
                                            cache_pipe_valid3 <= 1'b0;
                                            cache_pipe_valid4 <= 1'b0;
                                            cache_pipe_valid5 <= 1'b0;
                                            cache_pipe_last0 <= 1'b0;
                                            cache_pipe_last1 <= 1'b0;
                                            cache_pipe_last2 <= 1'b0;
                                            cache_pipe_last3 <= 1'b0;
                                            cache_pipe_last4 <= 1'b0;
                                            cache_pipe_last5 <= 1'b0;
                                            cache_pipe_active <= 1'b1;
                                            state <= ST_CACHE_PIPE;
                                        end else begin
                                            state <= ST_LOAD_X;
                                        end
                                    end
                                end else if ((packet_k < 4) || (packet_k > MAX_K)
                                    || (packet_k[1:0] != 2'b00)
                                    || (packet_rows < 1) || (packet_rows > MAX_ROWS)) begin
                                    protocol_error <= 1'b1;
                                    state <= ST_BATCH_HEADER;
                                end else begin
                                    active_k <= packet_k;
                                    active_rows <= packet_rows;
                                    activation_byte_index <= 1'b0;
                                    qbyte_index <= 2'd0;
                                    state <= ST_LOAD_X;
                                end
                            end
                        endcase
                    end

                    ST_LOAD_X: begin
                        if (!activation_byte_index) begin
                            if ((activation_count == {K_INDEX_W{1'b0}})
                                && (active_mode != MODE_CACHE_RUN)) begin
                                rows_remaining <= active_rows;
                                row_is_last <= (active_rows == 16'd1);
                            end
                            activation_byte_low <= input_data;
                            activation_byte_index <= 1'b1;
                        end else begin
                            if (active_mode == MODE_CACHE_RUN) begin
                                case (activation_count[1:0])
                                    2'd0: activation_stage0[activation_count[3:2] * 16 +: 16] <= $signed({input_data, activation_byte_low});
                                    2'd1: activation_stage1[activation_count[3:2] * 16 +: 16] <= $signed({input_data, activation_byte_low});
                                    2'd2: activation_stage2[activation_count[3:2] * 16 +: 16] <= $signed({input_data, activation_byte_low});
                                    default: activation_stage3[activation_count[3:2] * 16 +: 16] <= $signed({input_data, activation_byte_low});
                                endcase
                            end else begin
                                case (activation_count[1:0])
                                    2'd0: activation_bank0[activation_count[K_INDEX_W-1:2]] <= $signed({input_data, activation_byte_low});
                                    2'd1: activation_bank1[activation_count[K_INDEX_W-1:2]] <= $signed({input_data, activation_byte_low});
                                    2'd2: activation_bank2[activation_count[K_INDEX_W-1:2]] <= $signed({input_data, activation_byte_low});
                                    default: activation_bank3[activation_count[K_INDEX_W-1:2]] <= $signed({input_data, activation_byte_low});
                                endcase
                            end
                            activation_byte_index <= 1'b0;
                            if (activation_is_last) begin
                                activation_count <= {K_INDEX_W{1'b0}};
                                activation_is_last <= 1'b0;
                                row_index <= {ROW_INDEX_W{1'b0}};
                                group_index <= {GROUP_INDEX_W{1'b0}};
                                qbyte_index <= 2'd0;
                                accumulator <= 32'sd0;
                                if (active_mode == MODE_CACHE_RUN) begin
                                    cache_pipe_issue_done <= 1'b0;
                                    cache_pipe_valid0 <= 1'b0;
                                    cache_pipe_valid1 <= 1'b0;
                                    cache_pipe_valid2 <= 1'b0;
                                    cache_pipe_valid3 <= 1'b0;
                                    cache_pipe_valid4 <= 1'b0;
                                    cache_pipe_valid5 <= 1'b0;
                                    cache_pipe_last0 <= 1'b0;
                                    cache_pipe_last1 <= 1'b0;
                                    cache_pipe_last2 <= 1'b0;
                                    cache_pipe_last3 <= 1'b0;
                                    cache_pipe_last4 <= 1'b0;
                                    cache_pipe_last5 <= 1'b0;
                                    state <= ST_CACHE_PIPE;
                                    cache_pipe_active <= 1'b1;
                                end else begin
                                    state <= ST_LOAD_Q;
                                end
                            end else begin
                                activation_count <= activation_count + 1'b1;
                                activation_is_last <= (activation_count + 1'b1 == last_activation_index);
                            end
                        end
                    end

                    ST_LOAD_Q: begin
                        case (qbyte_index)
                            2'd0: begin
                                packed_byte0 <= input_data;
                                qbyte_index <= 2'd1;
                            end
                            2'd1: begin
                                packed_byte1 <= input_data;
                                qbyte_index <= 2'd2;
                            end
                            default: begin
                                qbyte_index <= 2'd0;
                                if (active_mode == MODE_CACHE_LOAD) begin
                                    cache_address <= cache_address + 1'b1;
                                    if (group_index == last_group_index) begin
                                        group_index <= {GROUP_INDEX_W{1'b0}};
                                         if (row_is_last) begin
                                            cache_valid[cache_slot] <= 1'b1;
                                             if (operation_index == last_operation_index) begin
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
                                            rows_remaining <= rows_remaining - 1'b1;
                                            row_is_last <= (rows_remaining == 16'd2);
                                            state <= ST_LOAD_Q;
                                        end
                                    end else begin
                                        group_index <= group_index + 1'b1;
                                        state <= ST_LOAD_Q;
                                    end
                                end else begin
                                    q0 <= $signed(packed_byte0[5:0]);
                                    q1 <= $signed({packed_byte1[3:0], packed_byte0[7:6]});
                                    q2 <= $signed({input_data[1:0], packed_byte1[7:4]});
                                    q3 <= $signed(input_data[7:2]);
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

            if (cache_pipe_active) begin
                if (CACHE_LANES == 32) begin
                    cache_pipe32_valid0 <= cache_pipe_issue;
                    cache_pipe32_valid1 <= cache_pipe32_valid0;
                    cache_pipe32_valid2 <= cache_pipe32_valid1;
                    cache_pipe32_valid3 <= cache_pipe32_valid2;
                    cache_pipe32_valid4 <= cache_pipe32_valid3;
                    cache_pipe32_valid5 <= cache_pipe32_valid4;
                    cache_pipe32_valid6 <= cache_pipe32_valid5;
                    cache_pipe32_valid7 <= cache_pipe32_valid6;
                    cache_pipe32_valid8 <= cache_pipe32_valid7;
                    cache_pipe32_valid9 <= cache_pipe32_valid8;
                    cache_pipe32_valid10 <= cache_pipe32_valid9;
                    cache_pipe32_last0 <= cache_pipe_issue && (group_index == CACHE_RUN_LAST_GROUP);
                    cache_pipe32_last1 <= cache_pipe32_last0;
                    cache_pipe32_last2 <= cache_pipe32_last1;
                    cache_pipe32_last3 <= cache_pipe32_last2;
                    cache_pipe32_last4 <= cache_pipe32_last3;
                    cache_pipe32_last5 <= cache_pipe32_last4;
                    cache_pipe32_last6 <= cache_pipe32_last5;
                    cache_pipe32_last7 <= cache_pipe32_last6;
                    cache_pipe32_last8 <= cache_pipe32_last7;
                    cache_pipe32_last9 <= cache_pipe32_last8;
                    cache_pipe32_last10 <= cache_pipe32_last9;

                    if (cache_pipe_issue) begin
                        cache_address <= cache_address + 18'd8;
                        if (group_index == CACHE_RUN_LAST_GROUP)
                            cache_pipe_issue_done <= 1'b1;
                        else
                            group_index <= group_index + 1'b1;
                    end

                    if (cache_pipe32_valid0) begin
                        q_stage0 <= $signed(cache_read_bank0[5:0]);
                        q_stage1 <= $signed(cache_read_bank0[11:6]);
                        q_stage2 <= $signed(cache_read_bank0[17:12]);
                        q_stage3 <= $signed(cache_read_bank0[23:18]);
                        q_stage4 <= $signed(cache_read_bank1[5:0]);
                        q_stage5 <= $signed(cache_read_bank1[11:6]);
                        q_stage6 <= $signed(cache_read_bank1[17:12]);
                        q_stage7 <= $signed(cache_read_bank1[23:18]);
                        q_stage8 <= $signed(cache_read_bank2[5:0]);
                        q_stage9 <= $signed(cache_read_bank2[11:6]);
                        q_stage10 <= $signed(cache_read_bank2[17:12]);
                        q_stage11 <= $signed(cache_read_bank2[23:18]);
                        q_stage12 <= $signed(cache_read_bank3[5:0]);
                        q_stage13 <= $signed(cache_read_bank3[11:6]);
                        q_stage14 <= $signed(cache_read_bank3[17:12]);
                        q_stage15 <= $signed(cache_read_bank3[23:18]);
                        q_stage16 <= $signed(cache_read_bank4[5:0]);
                        q_stage17 <= $signed(cache_read_bank4[11:6]);
                        q_stage18 <= $signed(cache_read_bank4[17:12]);
                        q_stage19 <= $signed(cache_read_bank4[23:18]);
                        q_stage20 <= $signed(cache_read_bank5[5:0]);
                        q_stage21 <= $signed(cache_read_bank5[11:6]);
                        q_stage22 <= $signed(cache_read_bank5[17:12]);
                        q_stage23 <= $signed(cache_read_bank5[23:18]);
                        q_stage24 <= $signed(cache_read_bank6[5:0]);
                        q_stage25 <= $signed(cache_read_bank6[11:6]);
                        q_stage26 <= $signed(cache_read_bank6[17:12]);
                        q_stage27 <= $signed(cache_read_bank6[23:18]);
                        q_stage28 <= $signed(cache_read_bank7[5:0]);
                        q_stage29 <= $signed(cache_read_bank7[11:6]);
                        q_stage30 <= $signed(cache_read_bank7[17:12]);
                        q_stage31 <= $signed(cache_read_bank7[23:18]);
                        x_stage0 <= activation_read_bank0[15:0];
                        x_stage1 <= activation_read_bank1[15:0];
                        x_stage2 <= activation_read_bank2[15:0];
                        x_stage3 <= activation_read_bank3[15:0];
                        x_stage4 <= activation_read_bank0[31:16];
                        x_stage5 <= activation_read_bank1[31:16];
                        x_stage6 <= activation_read_bank2[31:16];
                        x_stage7 <= activation_read_bank3[31:16];
                        x_stage8 <= activation_read_bank0[47:32];
                        x_stage9 <= activation_read_bank1[47:32];
                        x_stage10 <= activation_read_bank2[47:32];
                        x_stage11 <= activation_read_bank3[47:32];
                        x_stage12 <= activation_read_bank0[63:48];
                        x_stage13 <= activation_read_bank1[63:48];
                        x_stage14 <= activation_read_bank2[63:48];
                        x_stage15 <= activation_read_bank3[63:48];
                        x_stage16 <= activation_read_bank0_hi[15:0];
                        x_stage17 <= activation_read_bank1_hi[15:0];
                        x_stage18 <= activation_read_bank2_hi[15:0];
                        x_stage19 <= activation_read_bank3_hi[15:0];
                        x_stage20 <= activation_read_bank0_hi[31:16];
                        x_stage21 <= activation_read_bank1_hi[31:16];
                        x_stage22 <= activation_read_bank2_hi[31:16];
                        x_stage23 <= activation_read_bank3_hi[31:16];
                        x_stage24 <= activation_read_bank0_hi[47:32];
                        x_stage25 <= activation_read_bank1_hi[47:32];
                        x_stage26 <= activation_read_bank2_hi[47:32];
                        x_stage27 <= activation_read_bank3_hi[47:32];
                        x_stage28 <= activation_read_bank0_hi[63:48];
                        x_stage29 <= activation_read_bank1_hi[63:48];
                        x_stage30 <= activation_read_bank2_hi[63:48];
                        x_stage31 <= activation_read_bank3_hi[63:48];
                    end

                    if (cache_pipe32_valid1) begin
                        q0 <= q_stage0; q1 <= q_stage1; q2 <= q_stage2; q3 <= q_stage3;
                        q4 <= q_stage4; q5 <= q_stage5; q6 <= q_stage6; q7 <= q_stage7;
                        q8 <= q_stage8; q9 <= q_stage9; q10 <= q_stage10; q11 <= q_stage11;
                        q12 <= q_stage12; q13 <= q_stage13; q14 <= q_stage14; q15 <= q_stage15;
                        q16 <= q_stage16; q17 <= q_stage17; q18 <= q_stage18; q19 <= q_stage19;
                        q20 <= q_stage20; q21 <= q_stage21; q22 <= q_stage22; q23 <= q_stage23;
                        q24 <= q_stage24; q25 <= q_stage25; q26 <= q_stage26; q27 <= q_stage27;
                        q28 <= q_stage28; q29 <= q_stage29; q30 <= q_stage30; q31 <= q_stage31;
                        x0 <= x_stage0; x1 <= x_stage1; x2 <= x_stage2; x3 <= x_stage3;
                        x4 <= x_stage4; x5 <= x_stage5; x6 <= x_stage6; x7 <= x_stage7;
                        x8 <= x_stage8; x9 <= x_stage9; x10 <= x_stage10; x11 <= x_stage11;
                        x12 <= x_stage12; x13 <= x_stage13; x14 <= x_stage14; x15 <= x_stage15;
                        x16 <= x_stage16; x17 <= x_stage17; x18 <= x_stage18; x19 <= x_stage19;
                        x20 <= x_stage20; x21 <= x_stage21; x22 <= x_stage22; x23 <= x_stage23;
                        x24 <= x_stage24; x25 <= x_stage25; x26 <= x_stage26; x27 <= x_stage27;
                        x28 <= x_stage28; x29 <= x_stage29; x30 <= x_stage30; x31 <= x_stage31;
                    end

                    if (cache_pipe32_valid5) begin
                        pair_sum0_reg <= pair_sum0_next; pair_sum1_reg <= pair_sum1_next;
                        pair_sum2_reg <= pair_sum2_next; pair_sum3_reg <= pair_sum3_next;
                        pair_sum4_reg <= pair_sum4_next; pair_sum5_reg <= pair_sum5_next;
                        pair_sum6_reg <= pair_sum6_next; pair_sum7_reg <= pair_sum7_next;
                        pair_sum8_reg <= pair_sum8_next; pair_sum9_reg <= pair_sum9_next;
                        pair_sum10_reg <= pair_sum10_next; pair_sum11_reg <= pair_sum11_next;
                        pair_sum12_reg <= pair_sum12_next; pair_sum13_reg <= pair_sum13_next;
                        pair_sum14_reg <= pair_sum14_next; pair_sum15_reg <= pair_sum15_next;
                    end
                    if (cache_pipe32_valid6) begin
                        quartet0_reg <= quartet32_0_next; quartet1_reg <= quartet32_1_next;
                        quartet2_reg <= quartet32_2_next; quartet3_reg <= quartet32_3_next;
                        quartet4_reg <= quartet32_4_next; quartet5_reg <= quartet32_5_next;
                        quartet6_reg <= quartet32_6_next; quartet7_reg <= quartet32_7_next;
                    end
                    if (cache_pipe32_valid7) begin
                        cache_pipe32_sum8_0_reg <= cache_pipe32_sum8_0_next;
                        cache_pipe32_sum8_1_reg <= cache_pipe32_sum8_1_next;
                        cache_pipe32_sum8_2_reg <= cache_pipe32_sum8_2_next;
                        cache_pipe32_sum8_3_reg <= cache_pipe32_sum8_3_next;
                    end
                    if (cache_pipe32_valid8) begin
                        cache_pipe32_sum16_0_reg <= cache_pipe32_sum16_0_next;
                        cache_pipe32_sum16_1_reg <= cache_pipe32_sum16_1_next;
                    end
                    if (cache_pipe32_valid9)
                        cache_pipe32_group_sum_reg <= cache_pipe32_group_sum_next;

                    if (cache_pipe32_valid10) begin
                        accumulator <= cache_pipe32_accumulator_next;
                        if (cache_pipe32_last10) begin
                            m_axis_tdata <= cache_pipe32_accumulator_next;
                            m_axis_tvalid <= 1'b1;
                            m_axis_tlast <= (operation_index == last_operation_index)
                                          && row_is_last;
                            state <= ST_OUTPUT;
                            cache_pipe_active <= 1'b0;
                        end
                    end
                end else begin
                    cache_pipe_valid0 <= cache_pipe_issue;
                    cache_pipe_valid1 <= cache_pipe_valid0;
                    cache_pipe_valid2 <= cache_pipe_valid1;
                    cache_pipe_valid3 <= cache_pipe_valid2;
                    cache_pipe_valid4 <= cache_pipe_valid3;
                    cache_pipe_valid5 <= cache_pipe_valid4;
                    cache_pipe_last0 <= cache_pipe_issue && (group_index == last_group_index);
                    cache_pipe_last1 <= cache_pipe_last0;
                    cache_pipe_last2 <= cache_pipe_last1;
                    cache_pipe_last3 <= cache_pipe_last2;
                    cache_pipe_last4 <= cache_pipe_last3;
                    cache_pipe_last5 <= cache_pipe_last4;

                    if (cache_pipe_issue) begin
                        cache_address <= cache_address + 18'd4;
                        if (group_index == last_group_index)
                            cache_pipe_issue_done <= 1'b1;
                        else
                            group_index <= group_index + 1'b1;
                    end

                    if (cache_pipe_valid0) begin
                        q_stage0 <= $signed(cache_read_bank0[5:0]);
                        q_stage1 <= $signed(cache_read_bank0[11:6]);
                        q_stage2 <= $signed(cache_read_bank0[17:12]);
                        q_stage3 <= $signed(cache_read_bank0[23:18]);
                        q_stage4 <= $signed(cache_read_bank1[5:0]);
                        q_stage5 <= $signed(cache_read_bank1[11:6]);
                        q_stage6 <= $signed(cache_read_bank1[17:12]);
                        q_stage7 <= $signed(cache_read_bank1[23:18]);
                        q_stage8 <= $signed(cache_read_bank2[5:0]);
                        q_stage9 <= $signed(cache_read_bank2[11:6]);
                        q_stage10 <= $signed(cache_read_bank2[17:12]);
                        q_stage11 <= $signed(cache_read_bank2[23:18]);
                        q_stage12 <= $signed(cache_read_bank3[5:0]);
                        q_stage13 <= $signed(cache_read_bank3[11:6]);
                        q_stage14 <= $signed(cache_read_bank3[17:12]);
                        q_stage15 <= $signed(cache_read_bank3[23:18]);
                        x_stage0 <= activation_read_bank0[15:0];
                        x_stage1 <= activation_read_bank1[15:0];
                        x_stage2 <= activation_read_bank2[15:0];
                        x_stage3 <= activation_read_bank3[15:0];
                        x_stage4 <= activation_read_bank0[31:16];
                        x_stage5 <= activation_read_bank1[31:16];
                        x_stage6 <= activation_read_bank2[31:16];
                        x_stage7 <= activation_read_bank3[31:16];
                        x_stage8 <= activation_read_bank0[47:32];
                        x_stage9 <= activation_read_bank1[47:32];
                        x_stage10 <= activation_read_bank2[47:32];
                        x_stage11 <= activation_read_bank3[47:32];
                        x_stage12 <= activation_read_bank0[63:48];
                        x_stage13 <= activation_read_bank1[63:48];
                        x_stage14 <= activation_read_bank2[63:48];
                        x_stage15 <= activation_read_bank3[63:48];
                    end

                    if (cache_pipe_valid1) begin
                        q0 <= q_stage0; q1 <= q_stage1; q2 <= q_stage2; q3 <= q_stage3;
                        q4 <= q_stage4; q5 <= q_stage5; q6 <= q_stage6; q7 <= q_stage7;
                        q8 <= q_stage8; q9 <= q_stage9; q10 <= q_stage10; q11 <= q_stage11;
                        q12 <= q_stage12; q13 <= q_stage13; q14 <= q_stage14; q15 <= q_stage15;
                        x0 <= x_stage0; x1 <= x_stage1; x2 <= x_stage2; x3 <= x_stage3;
                        x4 <= x_stage4; x5 <= x_stage5; x6 <= x_stage6; x7 <= x_stage7;
                        x8 <= x_stage8; x9 <= x_stage9; x10 <= x_stage10; x11 <= x_stage11;
                        x12 <= x_stage12; x13 <= x_stage13; x14 <= x_stage14; x15 <= x_stage15;
                    end

                    if (cache_pipe_valid2) begin
                        product0_reg <= product0_next; product1_reg <= product1_next;
                        product2_reg <= product2_next; product3_reg <= product3_next;
                        product4_reg <= product4_next; product5_reg <= product5_next;
                        product6_reg <= product6_next; product7_reg <= product7_next;
                        product8_reg <= product8_next; product9_reg <= product9_next;
                        product10_reg <= product10_next; product11_reg <= product11_next;
                        product12_reg <= product12_next; product13_reg <= product13_next;
                        product14_reg <= product14_next; product15_reg <= product15_next;
                    end

                    if (cache_pipe_valid3) begin
                        quartet0_reg <= quartet0_next; quartet1_reg <= quartet1_next;
                        quartet2_reg <= quartet2_next; quartet3_reg <= quartet3_next;
                    end

                    if (cache_pipe_valid4)
                        cache_pipe_group_sum_reg <= cache_pipe_group_sum;

                    if (cache_pipe_valid5) begin
                        accumulator <= cache_pipe_accumulator_next;
                        if (cache_pipe_last5) begin
                            m_axis_tdata <= cache_pipe_accumulator_next;
                            m_axis_tvalid <= 1'b1;
                            m_axis_tlast <= (operation_index == last_operation_index)
                                          && row_is_last;
                            state <= ST_OUTPUT;
                            cache_pipe_active <= 1'b0;
                        end
                    end
                end
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
                if (CACHE_LANES == 32) begin
                    state <= ST_MAC_MULT_WAIT;
                end else begin
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
            end

            if (state == ST_MAC_MULT_WAIT)
                state <= ST_MAC_MULT_WAIT2;

            if (state == ST_MAC_MULT_WAIT2)
                state <= ST_MAC_SUM;

            if (state == ST_MAC_SUM) begin
                if ((active_mode == MODE_CACHE_RUN) && (CACHE_LANES != 32)) begin
                    quartet0_reg <= quartet0_next;
                    quartet1_reg <= quartet1_next;
                    quartet2_reg <= quartet2_next;
                    quartet3_reg <= quartet3_next;
                    state <= ST_MAC_MERGE;
                end else begin
                    pair_sum0_reg <= pair_sum0_next;
                    pair_sum1_reg <= pair_sum1_next;
                    state <= ST_MAC_SUM2;
                end
            end

            if (state == ST_MAC_SUM2) begin
                group_sum_reg <= group_sum_pair_next;
                state <= ST_MAC_ACC;
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
                     m_axis_tlast <= (operation_index == last_operation_index)
                                   && row_is_last;
                    state <= ST_OUTPUT;
                end else begin
                    group_index <= group_index + 1'b1;
                    if (active_mode == MODE_CACHE_RUN) begin
                        cache_pipe_issue_done <= 1'b0;
                        cache_pipe_valid0 <= 1'b0;
                        cache_pipe_valid1 <= 1'b0;
                        cache_pipe_valid2 <= 1'b0;
                        cache_pipe_valid3 <= 1'b0;
                        cache_pipe_valid4 <= 1'b0;
                        cache_pipe_valid5 <= 1'b0;
                        cache_pipe_last0 <= 1'b0;
                        cache_pipe_last1 <= 1'b0;
                        cache_pipe_last2 <= 1'b0;
                        cache_pipe_last3 <= 1'b0;
                        cache_pipe_last4 <= 1'b0;
                        cache_pipe_last5 <= 1'b0;
                        state <= ST_CACHE_PIPE;
                        cache_pipe_active <= 1'b1;
                    end else begin
                        state <= ST_LOAD_Q;
                    end
                end
            end

            if ((state == ST_OUTPUT) && m_axis_tready) begin
                m_axis_tvalid <= 1'b0;
                m_axis_tlast <= 1'b0;
                accumulator <= 32'sd0;
                group_index <= {GROUP_INDEX_W{1'b0}};
                qbyte_index <= 2'd0;
                if (row_is_last) begin
                    row_index <= {ROW_INDEX_W{1'b0}};
                    row_is_last <= 1'b0;
                    activation_count <= {K_INDEX_W{1'b0}};
                     if (operation_index == last_operation_index) begin
                        operation_index <= 16'd0;
                         batch_count <= {BATCH_COUNT_W{1'b0}};
                        batch_header_index <= 1'b0;
                        state <= ST_BATCH_HEADER;
                    end else begin
                        operation_index <= operation_index + 1'b1;
                        header_index <= 2'd0;
                        state <= ST_MATRIX_HEADER;
                    end
                end else begin
                    row_index <= row_index + 1'b1;
                    rows_remaining <= rows_remaining - 1'b1;
                    row_is_last <= (rows_remaining == 16'd2);
                    if (active_mode == MODE_CACHE_RUN) begin
                        cache_pipe_issue_done <= 1'b0;
                        cache_pipe_valid0 <= 1'b0;
                        cache_pipe_valid1 <= 1'b0;
                        cache_pipe_valid2 <= 1'b0;
                        cache_pipe_valid3 <= 1'b0;
                        cache_pipe_valid4 <= 1'b0;
                        cache_pipe_valid5 <= 1'b0;
                        cache_pipe_last0 <= 1'b0;
                        cache_pipe_last1 <= 1'b0;
                        cache_pipe_last2 <= 1'b0;
                        cache_pipe_last3 <= 1'b0;
                        cache_pipe_last4 <= 1'b0;
                        cache_pipe_last5 <= 1'b0;
                        state <= ST_CACHE_PIPE;
                        cache_pipe_active <= 1'b1;
                    end else begin
                        state <= ST_LOAD_Q;
                    end
                end
            end
        end
    end

endmodule

`default_nettype wire
