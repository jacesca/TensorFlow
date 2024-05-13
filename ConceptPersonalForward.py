"""
Forward Propagation Process
    is the process by which a neural network computes its output given an
    input. This is achieved by successively passing the input through all
    the layers of the network.

Each layer transforms its input data based on its weights, biases, and the
activation function, producing an output. This output becomes the input to
the next layer, and the process repeats until the final layer produces the
network's output.

Here's a step-by-step breakdown for our perceptron:
1. Hidden Layer 1: The raw inputs are passed into the first hidden layer
   (layer1), producing layer1_outputs.
2. Hidden Layer 2: The outputs from the first hidden layer become inputs
   for the second hidden layer (layer2), resulting in layer2_outputs.
3. Output Layer: Similarly, the outputs of the second hidden layer serve
   as inputs for the final output layer (layer3). The output of this layer
   is the output of the entire neural network.
"""
import random
from ConceptPersonalMultilayer import Perceptron as OldPerceptron


# Set the seed
random.seed(10)


# Perceptron Class. Inherited by the perceptron written in the precious chapter
class Perceptron(OldPerceptron):

    # Forward Propagation:
    # Pass the input through each layer and get the final output.
    def forward(self, inputs):
        # 1. Pass the inputs through the first hidden layer
        layer1_outputs = self.layer1.forward(inputs)
        # 2. Pass the outputs of the first hidden layer through the
        # second hidden layer
        layer2_outputs = self.layer2.forward(layer1_outputs)
        # 3. Pass the outputs of the second hidden layer through the
        # output layer
        return self.layer3.forward(layer2_outputs)


if __name__ == '__main__':
    # Create a perceptron with 2 inputs, 3 neurons per hidden layer
    # and 1 output
    perceptron = Perceptron(2, 3, 1)

    # Get the output of the perceptron for different xor operations
    xor_0_0 = round(perceptron.forward([0, 0])[0], 3)
    xor_0_1 = round(perceptron.forward([0, 1])[0], 3)
    xor_1_0 = round(perceptron.forward([1, 0])[0], 3)
    xor_1_1 = round(perceptron.forward([1, 1])[0], 3)

    # Print ouptut of the perceptron and its rounded value
    print('0 xor 0 =', xor_0_0, '=', round(xor_0_0), f'({0^0})')  # noqa
    print('0 xor 1 =', xor_0_1, '=', round(xor_0_1), f'({0^1})')  # noqa
    print('1 xor 0 =', xor_1_0, '=', round(xor_1_0), f'({1^0})')  # noqa
    print('1 xor 1 =', xor_1_1, '=', round(xor_1_1), f'({1^1})')  # noqa
