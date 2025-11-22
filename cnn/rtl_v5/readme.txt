pipeline_top: top module that orchestrate everything
row_conv: wraps 28 row_engine modules
row_engine: engine of row convolution
tranpose: transposes the output of row conv so that the intermediate values can be used to col convolution
col_conv: wraps 11 col_engine modules
col_engine: engine of col convolution
flatten: flattens the col_conv outputs from [10:0][10:0] to [120:0]
dense_layer_parallel: wraps all the dense neurons to work parallel
dense_neuron: neuron architecture that does the MAC operations
argmax: selects the value and predicts the number by finding teh highes value of that index  


Full CNN pipeline: RowConv -> Transpose -> ColConv -> Flatten -> Dense Layer1 (32 nodes) -> Dense Layer2 (10 nodes) -> Argmax -> Predicted No.

- 28 row engines in parallel
- 1 transpose block
- 11 column engines in parallel
- 1 flatten block
- 32 neurons working in parallel
- 10 neurons working in parallel 
- 1 argmax block 


