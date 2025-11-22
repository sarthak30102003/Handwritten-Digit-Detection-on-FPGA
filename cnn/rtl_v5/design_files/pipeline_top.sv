// =====================================================================
// pipeline_top.sv
// Full CNN-DNN pipeline (Row Conv -> Transpose -> Col Conv -> Flatten -> FC1 -> FC2 -> Argmax)
// =====================================================================

`timescale 1ns/1ps
module pipeline_top #(
    parameter int DATA_W = 16,
    parameter int IN_LEN = 28,
    parameter int K      = 8,
    parameter int OUT_W  = 11,
    parameter int ROWS   = 28,
    parameter int COLS   = 11
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done
);

    // ---------------------------------------------------------------
    // Shared kernel memories (populated by TB)
    // ---------------------------------------------------------------
    (* ram_style = "block" *) logic signed [DATA_W-1:0] row_kernel [0:K-1];
    (* ram_style = "block" *) logic signed [DATA_W-1:0] col_kernel [0:K-1];

    // ---------------------------------------------------------------
    // Interconnects
    // ---------------------------------------------------------------
    logic [ROWS-1:0] row_done;
    logic trans_done;
    logic [COLS-1:0] col_done;

    logic signed [DATA_W-1:0] row_out [0:ROWS-1][0:OUT_W-1];
    logic signed [DATA_W-1:0] trans_out [0:COLS-1][0:ROWS-1];
    logic signed [DATA_W-1:0] col_out [0:COLS-1][0:OUT_W-1];

    // Flatten layer I/O
    logic signed [DATA_W-1:0] flat_out [0:COLS*OUT_W-1];
    logic flatten_done;

    // ---------------------------------------------------------------
    // Row Convolution
    // ---------------------------------------------------------------
    row_conv #(
        .DATA_W(DATA_W), .IN_LEN(IN_LEN), .K(K),
        .OUT_W(OUT_W), .ROWS(ROWS)
    ) row_block (
        .clk(clk), .rst_n(rst_n), .start(start),
        .done(row_done),
        .out_values(row_out)
    );

    // ---------------------------------------------------------------
    // Transpose
    // ---------------------------------------------------------------
    transpose #(
        .DATA_W(DATA_W), .ROWS(ROWS), .COLS(OUT_W)
    ) tr (
        .clk(clk), .rst_n(rst_n), .start(&row_done), .done(trans_done),
        .in_row0 (row_out[0]),  .in_row1 (row_out[1]),
        .in_row2 (row_out[2]),  .in_row3 (row_out[3]),
        .in_row4 (row_out[4]),  .in_row5 (row_out[5]),
        .in_row6 (row_out[6]),  .in_row7 (row_out[7]),
        .in_row8 (row_out[8]),  .in_row9 (row_out[9]),
        .in_row10(row_out[10]), .in_row11(row_out[11]),
        .in_row12(row_out[12]), .in_row13(row_out[13]),
        .in_row14(row_out[14]), .in_row15(row_out[15]),
        .in_row16(row_out[16]), .in_row17(row_out[17]),
        .in_row18(row_out[18]), .in_row19(row_out[19]),
        .in_row20(row_out[20]), .in_row21(row_out[21]),
        .in_row22(row_out[22]), .in_row23(row_out[23]),
        .in_row24(row_out[24]), .in_row25(row_out[25]),
        .in_row26(row_out[26]), .in_row27(row_out[27]),
        .out_col0 (trans_out[0]),  .out_col1 (trans_out[1]),
        .out_col2 (trans_out[2]),  .out_col3 (trans_out[3]),
        .out_col4 (trans_out[4]),  .out_col5 (trans_out[5]),
        .out_col6 (trans_out[6]),  .out_col7 (trans_out[7]),
        .out_col8 (trans_out[8]),  .out_col9 (trans_out[9]),
        .out_col10(trans_out[10])
    );

    // ---------------------------------------------------------------
    // Column Convolution
    // ---------------------------------------------------------------
    col_conv #(
        .DATA_W(DATA_W), .IN_LEN(ROWS), .K(K),
        .OUT_W(OUT_W), .COLS(COLS)
    ) col_block (
        .clk(clk), .rst_n(rst_n), .start(trans_done),
        .in_cols(trans_out),
        .kernel(col_kernel),
        .done(col_done),
        .out_values(col_out)
    );
    
    // ---------------------------------------------------------------
    // Flatten Layer
    // ---------------------------------------------------------------
    flatten_layer #(
        .DATA_W(DATA_W),
        .COLS(COLS),
        .ROWS(OUT_W)
    ) flatten_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(&col_done),
        .done(flatten_done),
        .col_in(col_out),
        .flat_out(flat_out)
    );

    // ---------------------------------------------------------------
    // Dense Layer (121 Nodes)
    // ---------------------------------------------------------------
    logic dnn1_done, dnn2_done;
    logic signed [DATA_W-1:0] dense1_out [0:31];
    logic signed [DATA_W-1:0] dense2_out [0:9];
    logic [3:0] pred_class;

    dense_layer_parallel #(.DATA_W(DATA_W), .IN_NEUR(COLS*OUT_W), .OUT_NEUR(32)) dense1 (
        .clk(clk), .rst_n(rst_n), .start(flatten_done),
        .in_vec(flat_out), .out_vec(dense1_out), .done(dnn1_done)
    );
    
    // ---------------------------------------------------------------
    // Dense Layer (121 Nodes)
    // ---------------------------------------------------------------
    dense_layer_parallel #(.DATA_W(DATA_W), .IN_NEUR(32), .OUT_NEUR(10)) dense2 (
        .clk(clk), .rst_n(rst_n), .start(dnn1_done),
        .in_vec(dense1_out), .out_vec(dense2_out), .done(dnn2_done)
    );

    // ---------------------------------------------------------------
    // Argmax
    // ---------------------------------------------------------------
    argmax #(.DATA_W(DATA_W), .NUM_IN(10)) amx (
        .in_vec(dense2_out),
        .max_index(pred_class)
    );

    assign done = dnn2_done;

endmodule