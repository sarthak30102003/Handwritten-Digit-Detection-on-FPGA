// tb_pipeline_dense.sv
`timescale 1ns/1ps
module tb_pipeline_dense;
    parameter int DATA_W = 16;
    parameter int IN_LEN  = 28;
    parameter int K       = 8;
    parameter int OUT_W   = 11;
    parameter int ROWS    = 28;
    parameter int COLS    = 11;

    logic clk = 0;
    logic rst_n = 0;
    logic start = 0;
    logic done;

    pipeline_top #(
        .DATA_W(DATA_W), .IN_LEN(IN_LEN), .K(K),
        .OUT_W(OUT_W), .ROWS(ROWS), .COLS(COLS)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .done(done)
    );

    always #5 clk = ~clk; // 100 MHz clock

    reg signed [DATA_W-1:0] tmp_row [0:IN_LEN-1];
    reg signed [DATA_W-1:0] row_kernel [0:K-1];
    reg signed [DATA_W-1:0] col_kernel [0:K-1];
    reg signed [DATA_W-1:0] fc1_w  [0:121*32-1];
    reg signed [DATA_W-1:0] fc1_b  [0:31];
    reg signed [DATA_W-1:0] fc2_w  [0:32*10-1];
    reg signed [DATA_W-1:0] fc2_b  [0:9];

    integer idx;
    integer signed q;
    real v;

    initial begin
        $display("\n=== TB: Full CNN-DNN Pipeline Test ===");
        rst_n = 0; start = 0;
        #5; rst_n = 1;

        // -------------------------------------------------
        // Load convolution kernels
        // -------------------------------------------------
        $readmemh("row_conv_weights.mem", row_kernel);
        $readmemh("col_conv_weights.mem", col_kernel);
        for (int k = 0; k < K; k++) begin
            dut.row_kernel[k] = row_kernel[k];
            dut.col_kernel[k] = col_kernel[k];
        end

        // -------------------------------------------------
        // Hardcoded loading of 28 row input memories
        // (Corrected hierarchy below)
        // -------------------------------------------------
        $readmemh("row1.mem",  tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[0].row_engine_inst.input_mem[i]  = tmp_row[i];
        $readmemh("row2.mem",  tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[1].row_engine_inst.input_mem[i]  = tmp_row[i];
        $readmemh("row3.mem",  tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[2].row_engine_inst.input_mem[i]  = tmp_row[i];
        $readmemh("row4.mem",  tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[3].row_engine_inst.input_mem[i]  = tmp_row[i];
        $readmemh("row5.mem",  tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[4].row_engine_inst.input_mem[i]  = tmp_row[i];
        $readmemh("row6.mem",  tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[5].row_engine_inst.input_mem[i]  = tmp_row[i];
        $readmemh("row7.mem",  tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[6].row_engine_inst.input_mem[i]  = tmp_row[i];
        $readmemh("row8.mem",  tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[7].row_engine_inst.input_mem[i]  = tmp_row[i];
        $readmemh("row9.mem",  tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[8].row_engine_inst.input_mem[i]  = tmp_row[i];
        $readmemh("row10.mem", tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[9].row_engine_inst.input_mem[i]  = tmp_row[i];
        $readmemh("row11.mem", tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[10].row_engine_inst.input_mem[i] = tmp_row[i];
        $readmemh("row12.mem", tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[11].row_engine_inst.input_mem[i] = tmp_row[i];
        $readmemh("row13.mem", tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[12].row_engine_inst.input_mem[i] = tmp_row[i];
        $readmemh("row14.mem", tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[13].row_engine_inst.input_mem[i] = tmp_row[i];
        $readmemh("row15.mem", tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[14].row_engine_inst.input_mem[i] = tmp_row[i];
        $readmemh("row16.mem", tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[15].row_engine_inst.input_mem[i] = tmp_row[i];
        $readmemh("row17.mem", tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[16].row_engine_inst.input_mem[i] = tmp_row[i];
        $readmemh("row18.mem", tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[17].row_engine_inst.input_mem[i] = tmp_row[i];
        $readmemh("row19.mem", tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[18].row_engine_inst.input_mem[i] = tmp_row[i];
        $readmemh("row20.mem", tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[19].row_engine_inst.input_mem[i] = tmp_row[i];
        $readmemh("row21.mem", tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[20].row_engine_inst.input_mem[i] = tmp_row[i];
        $readmemh("row22.mem", tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[21].row_engine_inst.input_mem[i] = tmp_row[i];
        $readmemh("row23.mem", tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[22].row_engine_inst.input_mem[i] = tmp_row[i];
        $readmemh("row24.mem", tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[23].row_engine_inst.input_mem[i] = tmp_row[i];
        $readmemh("row25.mem", tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[24].row_engine_inst.input_mem[i] = tmp_row[i];
        $readmemh("row26.mem", tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[25].row_engine_inst.input_mem[i] = tmp_row[i];
        $readmemh("row27.mem", tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[26].row_engine_inst.input_mem[i] = tmp_row[i];
        $readmemh("row28.mem", tmp_row); for (int i=0;i<IN_LEN;i++) dut.row_block.RENG[27].row_engine_inst.input_mem[i] = tmp_row[i];

        // -------------------------------------------------
        // Dense Layer weight/bias loading
        // -------------------------------------------------
        $readmemh("fc1_weights.mem", fc1_w);
        $readmemh("fc1_bias.mem",    fc1_b);
        $readmemh("fc2_weights.mem", fc2_w);
        $readmemh("fc2_bias.mem",    fc2_b);

        idx = 0;
        for (int o = 0; o < 32; o++) begin
            for (int i = 0; i < 121; i++) begin
                dut.dense1.weights[o][i] = fc1_w[idx];
                idx++;
            end
            dut.dense1.bias[o] = fc1_b[o];
        end

        idx = 0;
        for (int o = 0; o < 10; o++) begin
            for (int i = 0; i < 32; i++) begin
                dut.dense2.weights[o][i] = fc2_w[idx];
                idx++;
            end
            dut.dense2.bias[o] = fc2_b[o];
        end

        // -------------------------------------------------
        // Start pipeline
        // -------------------------------------------------
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        // timeout guard
        fork
            begin
                wait(done === 1'b1);
            end
            begin
                #100000;
                $fatal("ERROR: Timeout waiting for done signal!");
            end
        join_any

        // -------------------------------------------------
        // Print results
        // -------------------------------------------------
        $display("\n=== Dense2 Outputs ===");
        for (int i = 0; i < 10; i++) begin
            q = dut.dense2_out[i];
            v = q / 256.0;
            $display("%0.8f",v);
        end

        $display("\nPredicted class = %0d", dut.pred_class);
        $display("=== TB: Simulation Complete ===\n");
        #100 $finish;
    end
endmodule
