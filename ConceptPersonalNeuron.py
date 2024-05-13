"""
The fundamental computational unit of a neural network is the neuron.
A neuron can be visualized as a small processing unit that takes multiple
inputs, processes them, and gives a single output.

Here's what happens step by step:
1. Each input is multiplied by a corresponding weight.
   The weights are learnable parameters and they determine the importance
   of the corresponding input.
2. All the weighted inputs are summed together.
3. In our implementation, we will also add an additional parameter called
   bias to the input sum. The bias allows the neuron to shift its output up
   or down, adding flexibility to the modeling capability.
4. Then the input sum is passed through an activation function. We are using
   the sigmoid function, which squashes values into the range (0, 1).
Note:
Bias of the neuron is also a trainable parameter.
"""
import random
import math

# Fix the seed of the "random" library, so it will be easier to test our code
# (random weights of the neuron will be the same every run)
random.seed(10)


# Sigmoid Activation Function: Used to introduce non-linearity into the model.
def sigmoid(x):
    # Returns the y value of sigmoid function for the given input (x value)
    return 1 / (1 + pow(math.e, -x))


class Neuron:
    def __init__(self, n_inputs):
        # n_inputs value describes how many inputs the neuron contains

        # Weights are initialized with random values.
        # We have here n_inputs neuron
        self.weights = [random.uniform(-1, 1) for _ in range(n_inputs)]

        # Bias is also initialized randomly.
        # Use the uniform function to generate a random bias for every neuron
        self.bias = random.uniform(-1, 1)

        # The resultant output of the neuron after processing the activation
        # function will be stored here.
        self.output = 0

        # The 'error' of the neuron, used during backpropagation.
        self.delta = 0

    def activate(self, inputs):
        # Used as a part of forward propagation

        # Calculate the sum of all inputs multiplied by its weights
        # Begin with input sum equals 0
        input_sum = 0
        for input_index, input_value in enumerate(inputs):
            # Iterate over every input
            # Add the result of multiplication of the input value by its
            # weight on every iteration
            input_sum += input_value * self.weights[input_index]

        # Add bias to the input sum
        input_sum_with_bias = input_sum + self.bias

        # Pass the input sum through the activation function
        self.output = sigmoid(input_sum_with_bias)
        return self.output


if __name__ == '__main__':
    # Test our neuron
    # Create a neuron with 3 inputs
    neuron = Neuron(3)
    # Generate inputs for the neuron
    neuron_inputs = [0.5, 0.2, -0.8]
    # Pass the inputs to the created neuron
    neuron_output = neuron.activate(neuron_inputs)

    print(f'Input of the neuron is {neuron_inputs}')
    print(f'Output of the neuron is {neuron_output:.3f}')
