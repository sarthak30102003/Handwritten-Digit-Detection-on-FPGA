// =====================================================================
// dense_neuron.sv
// A single neuron with internal MAC, parallel multiply accumulation
// =====================================================================
`timescale 1ns/1ps
module dense_neuron #(
    parameter int DATA_W = 16,
    parameter int IN_NEUR = 121
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    input  logic signed [DATA_W-1:0] in_vec [0:IN_NEUR-1],
    input  logic signed [DATA_W-1:0] weights [0:IN_NEUR-1],
    input  logic signed [DATA_W-1:0] bias,
    output logic signed [DATA_W-1:0] out_val,
    output logic done
);

    typedef enum logic [1:0] {IDLE, MAC, DONE} state_t;
    state_t state;

    integer idx;
    logic signed [47:0] acc;
    (* use_dsp = "yes" *) logic signed [31:0] prod;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            idx   <= 0;
            acc   <= 0;
            done  <= 0;
            out_val <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        idx <= 0;
                        acc <= 0;
                        state <= MAC;
                    end
                end

                MAC: begin
                    // Multiply and accumulate one input per cycle
                    prod = $signed(in_vec[idx]) * $signed(weights[idx]);
                    acc <= acc + $signed({{16{prod[31]}}, prod});
                    if (idx == IN_NEUR - 1)
                        state <= DONE;
                    else
                        idx <= idx + 1;
                end

                DONE: begin
                    // Add bias and quantize Q8.8
                    logic signed [47:0] acc_bias = acc + $signed({{32{bias[DATA_W-1]}}, bias});
                    logic signed [47:0] acc_rnd  = acc_bias + 48'sd128;
                    out_val <= acc_rnd[23:8];
                    done <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule
