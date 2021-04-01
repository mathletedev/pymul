import numpy as np


class Layer:
    def __init__(self):
        self.inputs = None
        self.outputs = None

    def forward_propagate(self, inputs):
        raise NotImplementedError

    def backward_propagate(self, errors, learning_rate):
        raise NotImplementedError
