`timescale 1ns/1ps
module argmax #(
    parameter int DATA_W = 16,
    parameter int NUM_IN = 10
)(
    input  logic signed [DATA_W-1:0] in_vec [0:NUM_IN-1],
    output logic [3:0] max_index
);
    integer i;
    logic signed [DATA_W-1:0] max_val;
    always_comb begin
        max_val = in_vec[0];
        max_index = 0;
        for (i = 1; i < NUM_IN; i++) begin
            if (in_vec[i] > max_val) begin
                max_val = in_vec[i];
                max_index = i[3:0];
            end
        end
    end
endmodule
