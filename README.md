# PyMuL

PyMuL is a machine learning library which includes neural networks and genetic algorithms!

## Installation

To install PyMuL, run:

```py
pip install pymul
```

## Usage

Here is a simple program that solves XOR:

```py
from pymul import Network


# Input data
# Use the Network.to_batch_array() function to convert to array
x_train = Network.to_batch_array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Expected outputs
y_train = Network.to_batch_array([[0], [1], [1], [0]])

# Initialize network with 2 input nodes, 1 layer of 3 hidden nodes, and 1 output node
network = Network([2, 3, 1])

# Train network
network.batch_train(x_train, y_train)

# Predict outputs
print(network.batch_predict(x_train))
```

## Credits

Thanks to Omar Aflak for his Neural Network from scratch in Python tutorial! It helped a lot while making this project!
Also, thanks to Coding Tech on YouTube for the tutorial on publishing Python packages!
