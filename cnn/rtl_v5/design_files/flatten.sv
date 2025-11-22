// =====================================================================
// flatten_layer.sv
// ----------------------------------------------------------------------------
// Flattens 11x11 col_out into 1x121 flat_out in column-major order
// flat_out[0]   = col_out[0][0]
// flat_out[1]   = col_out[1][0]
// ...
// flat_out[10]  = col_out[10][0]
// flat_out[11]  = col_out[0][1]
// ...
// flat_out[120] = col_out[10][10]
// =====================================================================
`timescale 1ns/1ps

module flatten_layer #(
    parameter int DATA_W = 16,
    parameter int COLS   = 11,
    parameter int ROWS   = 11
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done,

    input  logic signed [DATA_W-1:0] col_in [0:COLS-1][0:ROWS-1],
    output logic signed [DATA_W-1:0] flat_out [0:COLS*ROWS-1]
);

    typedef enum logic [1:0] {IDLE, COPY, DONE} state_t;
    state_t state;

    integer col_idx;
    integer row_idx;
    integer flat_idx;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= IDLE;
            col_idx   <= 0;
            row_idx   <= 0;
            flat_idx  <= 0;
            done      <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        col_idx  <= 0;
                        row_idx  <= 0;
                        flat_idx <= 0;
                        state    <= COPY;
                    end
                end

                COPY: begin
                    // true column-major order: loop over rows inside columns
                    flat_out[flat_idx] <= col_in[col_idx][row_idx];

                    if (row_idx == ROWS-1) begin
                        row_idx <= 0;
                        if (col_idx == COLS-1)
                            state <= DONE;
                        else
                            col_idx <= col_idx + 1;
                    end else begin
                        row_idx <= row_idx + 1;
                    end

                    flat_idx <= flat_idx + 1;
                end

                DONE: begin
                    done  <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
