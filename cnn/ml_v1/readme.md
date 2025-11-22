train.py : model training script (in signed Q8.8 fixed point formatting)
image_gen.py : takes a random image from mnist dataset and saves it as input.txt ( normalized from -1 to 1)
image_mem.py : converts the input.txt image to input.mem file (values stored in row major format)
inference.py : a test script to infer input.txt image from the trained weights (uses txt files of input and kernel, weights, baises) and saves the final prediction and intermediate products in layer_outputs folder.
dec2bin.py : converts the txt files of model_params folder of thier corresponding mem files in a new folder, model_params_mem.
bin2dec.py : finds the quantization errors and  its stats (min error, max error, avg error, median error, mode error) and the location of value which has max quantization error
split_mem.py : splits the input.mem file into 28 files row1.mem, row2.mem .... row28.mem which has values of each row in a new folder, split_mem_files.

8 kernels give max accuracy of 92%. Increasing or decreasing the number of kernels won't affect the accuracy much.

Order of execution : 
train.py -> dec2bin.py -> bin2dec.py -> image_gen.py -> inference.py -> image_mem.py -> split_mem.py

Training ML model using 8x8 trainable kernel.
Structure   : Input -> Row Conv -> Col Conv -> Fc layer (32) -> Output Layer -> Prediction
	
0th Layer   : Input Layer : 28*28 MNIST dataset

1st Layer   : Row Convolution Layer
    Input   : 28x28 MNIST Input Matrix
    Kernel  : 8 kernels with stride as 2 used for 1D convolution row wise
    Output  : 28x11 Output Matrix
    Size calculation : floor((Input Size - Kernel Size)/stride) + 1
                     : floor((28 - 8)/2) + 1 = 11

2nd Layer   : Col Convolution Layer
    Input   : 28x11 MNIST Input Matrix
    Kernel  : 8 kernels with stride as 2 used for 1D convolution col wise
    Output  : 11x11 Output Matrix
    Size calculation : floor((Input Size - Kernel Size)/stride) + 1
                     : floor((28 - 8)/2) + 1 = 11

3rd Layer   : Fully Connected Layer : Nodes : 32
    Input   : 11x11 Matrix (now flattened to 1D array i.e. 121 distinct values)
    Nodes   : 32
    Weights : 3872
    Weights Calculation  : Inputs * Nodes
                         : 121 * 32 = 3872
    Biases  : 32
    Biases Calculation   : = No. of Nodes = 32

4th Layer   : Output Layer : Nodes : 10
    Input   : 32
    Nodes   : 10
    Weights : 320
    Weights Calculation  : Inputs * Nodes
                         : 32 * 10 = 320
    Biases : 10
    Biases Calculation : = No. of Nodes = 10