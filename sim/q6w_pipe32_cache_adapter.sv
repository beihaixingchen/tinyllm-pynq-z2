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
    output wire [31:0] m_axis_tdata,
    output wire        m_axis_tvalid,
    input  wire        m_axis_tready,
    output wire        m_axis_tlast,
    output wire        protocol_error
);
    q6w_gemv_batched_pipe_axis #(
        .MAX_K(MAX_K),
        .MAX_ROWS(MAX_ROWS),
        .MAX_BATCH(MAX_BATCH),
        .CACHE_SLOTS(CACHE_SLOTS),
        .CACHE_WORDS(CACHE_WORDS),
        .CACHE_MATRIX_ROWS(CACHE_MATRIX_ROWS),
        .CACHE_LANES(32)
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
endmodule

`default_nettype wire
