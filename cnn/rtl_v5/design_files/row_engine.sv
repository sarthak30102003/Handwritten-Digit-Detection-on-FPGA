// engine.sv
`timescale 1ns/1ps
module row_engine #(
    parameter int DATA_W = 16,
    parameter int IN_LEN  = 28,
    parameter int K       = 8,
    parameter int OUT_W   = 11,
    parameter string INPUT_FILE  = "",
    parameter string KERNEL_FILE = ""
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done,

    output logic signed [DATA_W-1:0] out_values [0:OUT_W-1]
);

    (* keep = "true" *) (* ram_style = "block" *) logic signed [DATA_W-1:0] input_mem  [0:IN_LEN-1];
    (* keep = "true" *) (* ram_style = "block" *) logic signed [DATA_W-1:0] kernel_mem [0:K-1];

`ifndef SYNTHESIS
    initial begin
        $readmemh(INPUT_FILE,  input_mem);
        $readmemh(KERNEL_FILE, kernel_mem);
    end
`endif

    typedef enum logic [1:0] {IDLE, COMPUTE, STORE, DONE} state_t;
    state_t state, nstate;
    logic [5:0] base_idx;
    logic [3:0] out_idx;
    logic signed [31:0] m0,m1,m2,m3,m4,m5,m6,m7;
    logic signed [47:0] sum_full;
    logic signed [47:0] sum_reg;
    logic signed [15:0] out_q88;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; 
            base_idx <= 0; 
            out_idx <= 0; 
            done <= 0;
            
            for (int i = 0; i < OUT_W; i++) out_values[i] <= '0;
        end else begin
            state <= nstate;
            if (state == STORE) begin
                base_idx <= base_idx + 2;
                out_idx  <= out_idx + 1;
            end
            if (state == DONE) done <= 1;
        end
    end

    always_comb begin
        nstate = state;
        case (state)
            IDLE:    if (start) nstate = COMPUTE;
            COMPUTE: nstate = STORE;
            STORE:   nstate = (out_idx == OUT_W-1) ? DONE : COMPUTE;
            DONE:    nstate = DONE;
            default: nstate = IDLE;
        endcase
    end

    always_comb begin
        m0 = $signed(input_mem[base_idx+0]) * $signed(kernel_mem[0]);
        m1 = $signed(input_mem[base_idx+1]) * $signed(kernel_mem[1]);
        m2 = $signed(input_mem[base_idx+2]) * $signed(kernel_mem[2]);
        m3 = $signed(input_mem[base_idx+3]) * $signed(kernel_mem[3]);
        m4 = $signed(input_mem[base_idx+4]) * $signed(kernel_mem[4]);
        m5 = $signed(input_mem[base_idx+5]) * $signed(kernel_mem[5]);
        m6 = $signed(input_mem[base_idx+6]) * $signed(kernel_mem[6]);
        m7 = $signed(input_mem[base_idx+7]) * $signed(kernel_mem[7]);

        sum_full = $signed({{16{m0[31]}},m0}) +
                   $signed({{16{m1[31]}},m1}) +
                   $signed({{16{m2[31]}},m2}) +
                   $signed({{16{m3[31]}},m3}) +
                   $signed({{16{m4[31]}},m4}) +
                   $signed({{16{m5[31]}},m5}) +
                   $signed({{16{m6[31]}},m6}) +
                   $signed({{16{m7[31]}},m7});
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) sum_reg <= '0;
        else if (state == COMPUTE) sum_reg <= sum_full;
    end

    logic signed [47:0] sum_rounded;
    assign sum_rounded = sum_reg + 48'sd128;
    assign out_q88 = sum_rounded[23:8];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin end
        else if (state == STORE) begin
            case (out_idx)
                0:  out_values[0]  <= out_q88;
                1:  out_values[1]  <= out_q88;
                2:  out_values[2]  <= out_q88;
                3:  out_values[3]  <= out_q88;
                4:  out_values[4]  <= out_q88;
                5:  out_values[5]  <= out_q88;
                6:  out_values[6]  <= out_q88;
                7:  out_values[7]  <= out_q88;
                8:  out_values[8]  <= out_q88;
                9:  out_values[9]  <= out_q88;
                10: out_values[10] <= out_q88;
                default: ;
            endcase
        end
    end

endmodule
