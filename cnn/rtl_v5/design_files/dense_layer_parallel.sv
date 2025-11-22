// =====================================================================
// dense_layer_parallel.sv
// =====================================================================
`timescale 1ns/1ps
module dense_layer_parallel #(
    parameter int DATA_W   = 16,
    parameter int IN_NEUR  = 121,
    parameter int OUT_NEUR = 32
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    input  logic signed [DATA_W-1:0] in_vec [0:IN_NEUR-1],
    output logic signed [DATA_W-1:0] out_vec [0:OUT_NEUR-1],
    output logic done
);

    // Public memory for TB to load
    logic signed [DATA_W-1:0] weights [0:OUT_NEUR-1][0:IN_NEUR-1];
    logic signed [DATA_W-1:0] bias    [0:OUT_NEUR-1];

    logic [OUT_NEUR-1:0] done_flags;

    genvar n;
    generate
        for (n = 0; n < OUT_NEUR; n++) begin : NEUR
            dense_neuron #(.DATA_W(DATA_W), .IN_NEUR(IN_NEUR)) neuron_inst (
                .clk(clk),
                .rst_n(rst_n),
                .start(start),
                .in_vec(in_vec),
                .weights(weights[n]),
                .bias(bias[n]),
                .out_val(out_vec[n]),
                .done(done_flags[n])
            );
        end
    endgenerate

    assign done = &done_flags; 
endmodule
