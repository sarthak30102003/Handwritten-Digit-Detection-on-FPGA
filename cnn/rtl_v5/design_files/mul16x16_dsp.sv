// mul16x16_dsp.v
module mul16x16_dsp (
    input  wire         clk,
    input  wire signed [15:0] a,
    input  wire signed [15:0] b,
    output reg  signed [31:0] p
);

    // Register inputs
    reg signed [15:0] a_r;
    reg signed [15:0] b_r;

    always @(posedge clk) begin
        a_r <= a;
        b_r <= b;
    end

    // DSP multiply: Vivado guarantees DSP48E2 here
    (* use_dsp = "yes" *)
    always @(posedge clk) begin
        p <= a_r * b_r;
    end

endmodule
