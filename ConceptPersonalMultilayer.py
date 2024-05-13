"""
A multilayer perceptron (MLP).
Perceptron is the name of the simplest neural network, consisting of only one
hidden layer. However, in order to be able to solve more complex problems, we
will create a variation of perceptron called multilayer perceptron (MLP). A
multilayer perceptron consists of multiple hidden layers.

The structure of a multilayer perceptron looks like this:
1. An input layer: It receives the input data.
2. Hidden layers: These layers process the data and extract patterns.
   We have two hidden layers in our model.
3. Output layer: Produces the final prediction or classifications.
In general, each layer consists of multiple neurons, and the output from one
layer becomes the input for the next layer.
"""
# # If we have our code in AWS
# import os
# os.system('wget https://codefinity-content-media.s3.eu-west-1.amazonaws.com/f9fc718f-c98b-470d-ba78-d84ef16ba45f/2-2/prev_chapters_2.py 2>/dev/null')  # noqa
import random
from ConceptPersonalNeuron import Neuron


# Fix the seed for reproducible results
random.seed(10)


# Layer Class: Represents a layer of neurons.
class Layer:
    def __init__(self, n_neurons, n_inputs_per_neuron):
        # Create neurons and specify number of their inputs.
        self.neurons = [Neuron(n_inputs_per_neuron) for _ in range(n_neurons)]

    def forward(self, inputs):
        # Active neurons for forward propagation
        return [neuron.activate(inputs) for neuron in self.neurons]


# Perceptron Class: Represents the full multi-layer perceptron.
class Perceptron:
    def __init__(self, input_size, hidden_size, output_size):
        # Define three layers: 2 hidden layers and 1 output layer.
        self.layer1 = Layer(hidden_size, input_size)  # First hidden layer
        self.layer2 = Layer(hidden_size, hidden_size)  # Second hidden layer
        self.layer3 = Layer(output_size, hidden_size)  # Output layer


if __name__ == '__main__':
    # Create a perceptron with 2 inputs, 3 neurons per hidden layer and 1 output  # noqa
    perceptron = Perceptron(2, 3, 1)

    # Show weights of the third neuron in the second hidden layer
    print([round(weight, 2) for weight in perceptron.layer2.neurons[2].weights])  # noqa

    # Show weights of the single neuron in the output layer
    print([round(weight, 2) for weight in perceptron.layer3.neurons[0].weights])  # noqa
