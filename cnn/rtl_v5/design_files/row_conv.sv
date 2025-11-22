// =====================================================================
// row_conv.sv
// Wraps 28 row_engine instances
// =====================================================================
`timescale 1ns/1ps

module row_conv #(
    parameter int DATA_W = 16,
    parameter int IN_LEN = 28,
    parameter int K      = 8,
    parameter int OUT_W  = 11,
    parameter int ROWS   = 28
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start,

    // Packed output from all row engines
    output logic [ROWS-1:0] done,
    output logic signed [DATA_W-1:0] out_values [0:ROWS-1][0:OUT_W-1]
);

    genvar r;
    generate
        for (r = 0; r < ROWS; r++) begin : RENG
            row_engine #(
                .DATA_W(DATA_W),
                .IN_LEN(IN_LEN),
                .K(K),
                .OUT_W(OUT_W),
                .INPUT_FILE($sformatf("row%0d.mem", r+1)),
                .KERNEL_FILE("row_conv_weights.mem")
            ) row_engine_inst (
                .clk(clk),
                .rst_n(rst_n),
                .start(start),
                .done(done[r]),
                .out_values(out_values[r])
            );
        end
    endgenerate

endmodule
