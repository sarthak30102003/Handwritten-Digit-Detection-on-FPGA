#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
import matplotlib.pyplot
import csv
import os
import xlsxwriter


# In[2]:


def convert (imgs, labels, outfile, n):
    imgf = open(imgs, "rb")
    labelf = open(labels, "rb")
    csvf = open(outfile, "w")

    imgf.read(16)
    labelf.read(8)
    images = []

    for i in range (n):
        image = [ord(labelf.read(1))]
        for j in range (28*28):
            image.append(ord(imgf.read(1)))
        images.append(image)

    for image in images:
        csvf.write(",".join(str(pix) for pix in image) + "\n")
    imgf.close()
    labelf.close()
    csvf.close()


# In[3]:


mnist_train_x = "train-images.idx3-ubyte"
mnist_train_y = "train-labels.idx1-ubyte"
mnist_test_x = "t10k-images.idx3-ubyte"
mnist_test_y = "t10k-labels.idx1-ubyte"


# In[4]:


convert(mnist_train_x, mnist_train_y,"train.csv", 60000)
convert(mnist_test_x, mnist_test_y,"test.csv", 10000)


# In[5]:


train_file = open("train.csv", "r")
train_list = train_file.readlines()
train_file.close()
print(len(train_list))


# In[6]:


train_list[100]


# In[7]:


values = train_list[500].split(",")
image_array = np.asarray(values[1:], dtype=np.uint8).reshape(28, 28)  
matplotlib.pyplot.imshow(image_array, cmap="Greys", interpolation="None")


# In[8]:


test_file = open("test.csv", "r")
test_list = test_file.readlines()
test_file.close()
print(len(test_list))


# In[9]:


def convert_csv_to_single_mem(input_csv_path, output_mem_path, with_labels=True):
    with open(input_csv_path, 'r') as f, open(output_mem_path, 'w') as out:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            label = int(row[0])
            pixels = map(int, row[1:])

            if with_labels:
                out.write(f"// Image {idx}, Label {label}\n")

            for val in pixels:
                bin_val = format(val, '08b')  # Convert to 8-bit binary
                out.write(bin_val + "\n")

            # Optional separator between images
            # out.write("\n")

            if idx % 1000 == 0:
                print(f"Processed {idx} images...")

    print(f"✅ All image pixel data written to {output_mem_path}")


# In[10]:


class DNN:
    def __init__(self, sizes=[784, 128, 64, 10], epochs=10, learning_rate=0.01):
        self.sizes = sizes
        self.epochs = epochs
        self.learning_rate = learning_rate

        input_layer = sizes[0]
        hidden_1 = sizes[1]
        hidden_2 = sizes[2]
        output_layer = sizes[3]

        self.params = {
                'W1': np.random.randn(hidden_1, input_layer) * np.sqrt(1.0 / hidden_1),     # 128 * 784
                'W2': np.random.randn(hidden_2, hidden_1) * np.sqrt(1.0 / hidden_2),        # 64 * 128
                'W3': np.random.randn(output_layer, hidden_2) * np.sqrt(1.0 / output_layer) # 10 * 64
        }
        # self.weights = [np.random.randn(y, x) for x, y in zip(sizes)]

    def relu (self, x, derivative=False):
        if derivative:
            return np.where(x > 0, 1, 0)
        return np.maximum(0, x)

    def softmax(self, x, derivative=False):
        exps = np.exp(x-x.max())
        if derivative:
            return exps / np.sum(exps, axis = 0) * (1 - exps / np.sum(exps, axis = 0))
        return exps / np.sum(exps, axis = 0)

    def forward_pass(self, x_train):
        # This is a placeholder for the forward pass of the network
        params = self.params
        params['A0'] = x_train  # 784 * 1

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params['W1'], params['A0']) # 128 * 1
        params['A1'] = self.relu(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params['W2'], params['A1']) # 64 * 1
        params['A2'] = self.relu(params['Z2'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params['W3'], params['A2']) # 10 * 1
        params['A3'] = self.softmax(params['Z3'])

        return params['A3']

    def backward_pass(self, y_train, output):
        # This is a placeholder for the backward pass of the network
        params = self.params
        change_w = {}

        # calculate w3 update
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z3'], derivative=True)
        change_w['W3'] = np.outer(error, params['A2'])

        # calculate w2 update
        error = np.dot(params['W3'].T, error) * self.relu(params['Z2'], derivative=True)
        change_w['W2'] = np.outer(error, params['A1'])

        # calculate w1 update
        error = np.dot(params['W2'].T, error) * self.relu(params['Z1'], derivative=True)
        change_w['W1'] = np.outer(error, params['A0'])

        return change_w

    def update_weights(self, change_w, learning_rate):
        # This is a placeholder for the weight update of the network
        for key, val in change_w.items():
            self.params[key] = self.params[key] - self.learning_rate * val

    def compute_accuracy(self, test_data):
        # This is a placeholder for the accuracy computation of the network
        predictions = []
        for x in test_data:
                values = x.split(",")
                inputs = (np.asarray(values[1:], dtype=np.float32) / 255.0 * 0.99) + 0.01
                targets = np.zeros(10) + 0.01
                targets[int(values[0])] = 0.99
                output = self.forward_pass(inputs)
                pred = np.argmax(output)
                predictions.append(pred == np.argmax(targets))

        return np.mean(predictions)       

    def train(self, train_list, test_list):
        # This is a placeholder for the training of the network
        start_time = time.time()
        for i in range (self.epochs):
            for x in train_list:
                values = x.split(",")
                inputs = (np.asarray(values[1:], dtype=np.float32) / 255.0 * 0.99) + 0.01
                targets = np.zeros(10) + 0.01
                targets[int(values[0])] = 0.99
                output = self.forward_pass(inputs)
                change_w = self.backward_pass(targets, output)
                self.update_weights(change_w, self.learning_rate)

            accuracy = self.compute_accuracy(test_list)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
              i+1, time.time() - start_time, accuracy * 100
          ))

    def export_weights_to_mem(self, base_filename="W"):
        for i, key in enumerate(['W1', 'W2', 'W3'], start=1):
            weights = self.params[key]
            mem_filename = f"{base_filename}{i}.mem"
            csv_filename = f"{base_filename}{i}.csv"

            # Write binary .mem file (32-bit IEEE 754)
            with open(mem_filename, "w") as mem_file:
                for row in weights:
                    for weight in row:
                        fp32 = np.float32(weight)
                        binary_string = format(np.frombuffer(fp32.tobytes(), dtype=np.uint32)[0], '032b')
                        mem_file.write(binary_string + "\n")

            # Write decimal .csv file
            with open(csv_filename, "w", newline='') as csv_file:
                writer = csv.writer(csv_file)
                for row in weights:
                    writer.writerow([f"{val:.8f}" for val in row])  # Up to 8 decimal places

            print(f"✅ Exported {key} to {mem_filename} (binary) and {csv_filename} (decimal)")

    def export_layerwise_detail_to_excel(self, base_filename="dnn_layer"):
        import xlsxwriter

        if not all(k in self.params for k in ['A0', 'A1', 'A2', 'A3']):
            print("⚠️ Run a forward pass before exporting layer-wise details.")
            return

        headers = ["Neuron no.", "X", "w", "z", "OUT"]

        # Define each layer with its inputs, weights, z, and output
        layers = [
            (f"{base_filename}1.xlsx", self.params['A0'], self.params['W1'], self.params['Z1'], self.params['A1']),
            (f"{base_filename}2.xlsx", self.params['A1'], self.params['W2'], self.params['Z2'], self.params['A2']),
            (f"{base_filename}3.xlsx", self.params['A2'], self.params['W3'], self.params['Z3'], self.params['A3']),
        ]

        for filename, x, w, z, out in layers:
            workbook = xlsxwriter.Workbook(filename)
            sheet = workbook.add_worksheet("LayerDetails")
            sheet.write_row(0, 0, headers)

            for i in range(w.shape[0]):
                x_vals = ", ".join([f"{val:.5f}" for val in x])
                w_vals = ", ".join([f"{val:.5f}" for val in w[i]])
                z_val = f"{z[i]:.5f}"
                out_val = f"{out[i]:.5f}"
                sheet.write_row(i + 1, 0, [i, x_vals, w_vals, z_val, out_val])

            workbook.close()
            print(f"✅ Exported {filename}")

        def normalize_output_minus1_to_1(self, output):
            min_val = np.min(output)
            max_val = np.max(output)
            if max_val - min_val == 0:
                return np.zeros_like(output)
            norm_output = 2 * ((output - min_val) / (max_val - min_val)) - 1
            return norm_output


# In[ ]:


dnn = DNN(sizes=[784, 128, 64, 10], epochs=100, learning_rate=0.01)
dnn.train(train_list, test_list)


# In[31]:


# dnn.export_all_parameters_to_csv()
dnn.export_layerwise_detail_to_excel("dnn_layer")

