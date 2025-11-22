// transpose.sv  (widened transpose: parameter T_W, keeps original ports)
// - Keeps the same 2-cycle READ/WRITE handshake you had (so integration unchanged)
// - Default T_W = 11: copy 11 elements per READ/WRITE pair (one whole column-chunk)
// - This reduces cycles from 617 --> 57 (for ROWS=28, COLS=11, T_W=11)
// =====================================================================
`timescale 1ns/1ps
module transpose #(
    parameter int DATA_W = 16,
    parameter int ROWS   = 28,
    parameter int COLS   = 11,
    parameter int T_W    = 11   // widening factor (how many columns processed per chunk)
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done,

    input  logic signed [DATA_W-1:0] in_row0  [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row1  [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row2  [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row3  [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row4  [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row5  [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row6  [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row7  [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row8  [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row9  [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row10 [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row11 [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row12 [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row13 [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row14 [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row15 [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row16 [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row17 [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row18 [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row19 [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row20 [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row21 [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row22 [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row23 [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row24 [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row25 [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row26 [0:COLS-1],
    input  logic signed [DATA_W-1:0] in_row27 [0:COLS-1],

    output logic signed [DATA_W-1:0] out_col0  [0:ROWS-1],
    output logic signed [DATA_W-1:0] out_col1  [0:ROWS-1],
    output logic signed [DATA_W-1:0] out_col2  [0:ROWS-1],
    output logic signed [DATA_W-1:0] out_col3  [0:ROWS-1],
    output logic signed [DATA_W-1:0] out_col4  [0:ROWS-1],
    output logic signed [DATA_W-1:0] out_col5  [0:ROWS-1],
    output logic signed [DATA_W-1:0] out_col6  [0:ROWS-1],
    output logic signed [DATA_W-1:0] out_col7  [0:ROWS-1],
    output logic signed [DATA_W-1:0] out_col8  [0:ROWS-1],
    output logic signed [DATA_W-1:0] out_col9  [0:ROWS-1],
    output logic signed [DATA_W-1:0] out_col10 [0:ROWS-1]
);

    // basic safety: clamp T_W to [1,COLS]
    localparam int W = (T_W < 1) ? 1 : ((T_W > COLS) ? COLS : T_W);
    localparam int NUM_CHUNKS = (COLS + W - 1) / W;

    typedef enum logic [1:0] {IDLE, READ, WRITE, DONE} state_t;
    state_t state, nstate;

    // small indices as regs (use integers for easier arithmetic)
    integer row_idx;
    integer chunk_idx;
    integer k;
    integer col;

    // buffer to hold up to W items read during READ cycle
    logic signed [DATA_W-1:0] data_vec [0:W-1];

    // initialize outputs on reset
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= IDLE;
            row_idx   <= 0;
            chunk_idx <= 0;
            done      <= 1'b0;

            for (int r = 0; r < ROWS; r++) begin
                out_col0[r]  <= '0; out_col1[r]  <= '0; out_col2[r]  <= '0;
                out_col3[r]  <= '0; out_col4[r]  <= '0; out_col5[r]  <= '0;
                out_col6[r]  <= '0; out_col7[r]  <= '0; out_col8[r]  <= '0;
                out_col9[r]  <= '0; out_col10[r] <= '0;
            end
        end else begin
            state <= nstate;

            if (state == IDLE) begin
                // nothing
            end

            if (state == DONE) begin
                done <= 1'b1;
            end
        end
    end

    // next-state logic (keeps same handshake pattern: READ -> WRITE -> READ ...)
    always_comb begin
        nstate = state;
        case (state)
            IDLE:   if (start) nstate = READ;
            READ:   nstate = WRITE;
            WRITE:  nstate = ((chunk_idx == NUM_CHUNKS-1) && (row_idx == ROWS-1)) ? DONE : READ;
            DONE:   nstate = DONE;
            default: nstate = IDLE;
        endcase
    end

    // READ: capture up to W elems from the selected input row/columns into data_vec
    // WRITE: write data_vec entries into appropriate out_colX[row_idx]
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // already handled above
        end else begin
            if (state == READ) begin
                // compute base column index for this chunk
                automatic int base_col = chunk_idx * W;
                // for k in 0..W-1 capture from in_row<row_idx>[base_col+k] if valid
                for (k = 0; k < W; k++) begin
                    col = base_col + k;
                    if (col < COLS) begin
                        // select correct in_rowN based on row_idx
                        case (row_idx)
                            0:  data_vec[k] <= in_row0[col];
                            1:  data_vec[k] <= in_row1[col];
                            2:  data_vec[k] <= in_row2[col];
                            3:  data_vec[k] <= in_row3[col];
                            4:  data_vec[k] <= in_row4[col];
                            5:  data_vec[k] <= in_row5[col];
                            6:  data_vec[k] <= in_row6[col];
                            7:  data_vec[k] <= in_row7[col];
                            8:  data_vec[k] <= in_row8[col];
                            9:  data_vec[k] <= in_row9[col];
                            10: data_vec[k] <= in_row10[col];
                            11: data_vec[k] <= in_row11[col];
                            12: data_vec[k] <= in_row12[col];
                            13: data_vec[k] <= in_row13[col];
                            14: data_vec[k] <= in_row14[col];
                            15: data_vec[k] <= in_row15[col];
                            16: data_vec[k] <= in_row16[col];
                            17: data_vec[k] <= in_row17[col];
                            18: data_vec[k] <= in_row18[col];
                            19: data_vec[k] <= in_row19[col];
                            20: data_vec[k] <= in_row20[col];
                            21: data_vec[k] <= in_row21[col];
                            22: data_vec[k] <= in_row22[col];
                            23: data_vec[k] <= in_row23[col];
                            24: data_vec[k] <= in_row24[col];
                            25: data_vec[k] <= in_row25[col];
                            26: data_vec[k] <= in_row26[col];
                            27: data_vec[k] <= in_row27[col];
                            default: data_vec[k] <= '0;
                        endcase
                    end else begin
                        data_vec[k] <= '0;
                    end
                end
            end

            if (state == WRITE) begin
                // write data_vec entries into appropriate out_col<col>[row_idx]
                automatic int base_col = chunk_idx * W;
                for (k = 0; k < W; k++) begin
                    col = base_col + k;
                    if (col < COLS) begin
                        case (col)
                            0:  out_col0[row_idx]  <= data_vec[k];
                            1:  out_col1[row_idx]  <= data_vec[k];
                            2:  out_col2[row_idx]  <= data_vec[k];
                            3:  out_col3[row_idx]  <= data_vec[k];
                            4:  out_col4[row_idx]  <= data_vec[k];
                            5:  out_col5[row_idx]  <= data_vec[k];
                            6:  out_col6[row_idx]  <= data_vec[k];
                            7:  out_col7[row_idx]  <= data_vec[k];
                            8:  out_col8[row_idx]  <= data_vec[k];
                            9:  out_col9[row_idx]  <= data_vec[k];
                            10: out_col10[row_idx] <= data_vec[k];
                            default: ;
                        endcase
                    end
                end

                // advance indices after write
                if (chunk_idx == NUM_CHUNKS-1) begin
                    chunk_idx <= 0;
                    if (row_idx == ROWS-1) row_idx <= 0;
                    else row_idx <= row_idx + 1;
                end else begin
                    chunk_idx <= chunk_idx + 1;
                end
            end

            // when we enter IDLE from reset or after DONE, ensure indices are cleared on next start
            if (state == IDLE && start) begin
                row_idx   <= 0;
                chunk_idx <= 0;
                done      <= 1'b0;
            end
        end
    end

endmodule
