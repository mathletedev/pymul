from pymul.layers.layer import Layer


class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagate(self, inputs):
        self.inputs = inputs
        self.outputs = self.activation(self.inputs)

        return self.outputs

    def backward_propagate(self, errors, learning_rate):
        return self.activation_prime(self.inputs) * errors
