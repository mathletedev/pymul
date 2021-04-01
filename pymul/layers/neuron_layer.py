import numpy as np
from pymul.layers.layer import Layer


class NeuronLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) * 2 - 1
        self.biases = np.random.rand(1, output_size) * 2 - 1

    def forward_propagate(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(self.inputs, self.weights) + self.biases

        return self.outputs

    def backward_propagate(self, errors, learning_rate):
        input_errors = np.dot(errors, self.weights.T)
        weight_errors = np.dot(self.inputs.T, errors)

        self.weights -= weight_errors * learning_rate
        self.biases -= errors * learning_rate

        return input_errors
