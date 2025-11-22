// =====================================================================
// col_conv.sv
// Wraps 11 col_engine instances
// =====================================================================
`timescale 1ns/1ps

module col_conv #(
    parameter int DATA_W = 16,
    parameter int IN_LEN = 28,
    parameter int K      = 8,
    parameter int OUT_W  = 11,
    parameter int COLS   = 11
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start,

    input  logic signed [DATA_W-1:0] in_cols  [0:COLS-1][0:IN_LEN-1],
    input  logic signed [DATA_W-1:0] kernel   [0:K-1],

    output logic [COLS-1:0] done,
    output logic signed [DATA_W-1:0] out_values [0:COLS-1][0:OUT_W-1]
);

    genvar c;
    generate
        for (c = 0; c < COLS; c++) begin : CENG
            col_engine #(
                .DATA_W(DATA_W),
                .IN_LEN(IN_LEN),
                .OUT_W(OUT_W),
                .K(K)
            ) col_engine_inst (
                .clk(clk),
                .rst_n(rst_n),
                .start(start),
                .input_col(in_cols[c]),
                .kernel_in(kernel),
                .out_values(out_values[c]),
                .done(done[c])
            );
        end
    endgenerate

endmodule
