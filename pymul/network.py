import numpy as np

from pymul.functions.activation import tanh, tanh_prime
from pymul.functions.loss import mse, mse_prime
from pymul.layers.activation_layer import ActivationLayer
from pymul.layers.neuron_layer import NeuronLayer


class Network:
    def __init__(
        self,
        layer_sizes,
        activation=tanh,
        activation_prime=tanh_prime,
        loss=mse,
        loss_prime=mse_prime,
    ):
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            self.add_layer(NeuronLayer(layer_sizes[i], layer_sizes[i + 1]))
            self.add_layer(ActivationLayer(activation, activation_prime))

        self.loss = loss
        self.loss_prime = loss_prime

    def add_layer(self, layer):
        self.layers.append(layer)

    def predict(self, inputs):
        outputs = inputs

        for layer in self.layers:
            outputs = layer.forward_propagate(outputs)

        return outputs.tolist()[0]

    def batch_predict(self, batch_inputs):
        batch_outputs = []

        for inputs in batch_inputs:
            batch_outputs.append(self.predict(inputs))

        return batch_outputs

    def train(self, inputs, targets, learning_rate=0.1):
        outputs = self.predict(inputs)

        print("Error: %f" % self.loss(targets, outputs))

        errors = self.loss_prime(targets, outputs)
        for layer in reversed(self.layers):
            errors = layer.backward_propagate(errors, learning_rate)

    def batch_train(self, batch_inputs, batch_targets, epochs=1000, learning_rate=0.1):
        samples = len(batch_inputs)

        for i in range(epochs):
            total_error = 0

            for j in range(samples):
                outputs = self.predict(batch_inputs[j])

                total_error += self.loss(batch_targets[j], outputs)

                errors = self.loss_prime(batch_targets[j], outputs)
                for layer in reversed(self.layers):
                    errors = layer.backward_propagate(errors, learning_rate)

            total_error /= samples
            print("Epoch %d of %d | Error: %f" % (i + 1, epochs, total_error))

    @staticmethod
    def to_array(list):
        return np.array([list])

    @staticmethod
    def to_batch_array(list):
        return [np.array([i]) for i in list]

    @staticmethod
    def functions():
        return {
            "tanh": tanh,
            "tanh_prime": tanh_prime,
            "mse": mse,
            "mse_prime": mse_prime,
        }
