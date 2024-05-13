"""
Backpropagation is the most confusing part of neural network training.
At its core, it uses the gradient descent algorithm, which requires a
good understanding of calculus.
However, if you're interested in exploring the mathematical components
more thoroughly, you're welcome to enroll in the Mathematics for Data
Analysis and Modeling course.
We can split backpropagation algorithm into several steps:
1. Forward Propagation:
    At this step we pass our inputs through the perceptron to store
    outputs (Neuron.output) of every neuron.

2. Error Computing:
    In this phase, we determine the individual error for each neuron.
    This error indicates the difference between the neuron's output
    and the desired output.
    For neurons in the output layer, this is straightforward: when
    given a specific input, the error represents the difference between
    the neural network's prediction and the actual target value.
    For neurons in the hidden layers, the error measures the variation
    between their current output and the expected input for the subsequent
    layer.

3. Calculating the Gradient (Delta):
    At this stage, we calculate the degree and direction of each neuron's
    deviation. We achieve this by multiplying the neuron's error with the
    derivative of its activation function (in this case, sigmoid) based on
    its output.

    This computation should be executed concurrently with error calculation,
    as the current layer's gradient (delta) is essential for determining the
    error in the preceding layer. It also causes this process to be done in
    order from output layer to input layer (backward direction).

4. Modifying Weights and Biases (Taking a Step in Gradient Descent):
    The last step of the backpropagation process involves updating the
    neurons' weights and biases according to their respective deltas.

Note
Error computing and calculating the gradient should progress in reverse
order, moving from the output layer towards the input layer.

Learning Rate
    Another crucial aspect of model training is the learning rate. As an
    integral component of the gradient descent algorithm, the learning
    rate can be visualized as the pace of training.

    A higher learning rate accelerates the training process; however, an
    excessively high rate might cause the neural network to overlook
    valuable insights and patterns within the data.

    Note
        The learning rate is a floating point value between 0 and 1 and
        its used on the last step of the backpropagation algorithm to
        reduce the adjustments applied to the weights and biases.
        Selecting an optimal learning rate involves various methods
        known as hyperparameter tuning.

Epochs
    Every time our perceptron processes the entire dataset, we refer
    to it as an epoch. To effectively recognize patterns in the data,
    it's essential to feed our entire dataset into the model multiple
    times.

    We can utilize the XOR example as a validation test to ensure our
    [model is set up correctly. The XOR has only four unique combinations,
    all of which are derived from the truth table discussed in the preceding
    chapter.
    By training our neural network using these examples over 10,000 epochs
    and a learning rate of 0.2, we ensure the model comprehends the data.
"""
from ConceptPersonalForward import Perceptron as OldPerceptron  # noqa
from ConceptPersonalNeuron import sigmoid


# Derivative of the Sigmoid Function: Used during backpropagation.
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Perceptron Class.
class Perceptron(OldPerceptron):

    def fit(self, training_data, labels, epochs, learning_rate):
        # Training using the backpropagation algorithm.
        # Iterate over a number of epochs
        for epoch in range(epochs):
            # Iterate over every input example in training data
            for inputs, label in zip(training_data, labels):
                # 1. Run forward propagation.
                outputs = self.forward(inputs)

                # Output layer: Compute deltas (errors) for each neuron.
                for i, neuron in enumerate(self.layer3.neurons):
                    error = label[i] - outputs[i]
                    neuron.delta = error * sigmoid_derivative(neuron.output)

                # Hidden layer 2: Compute deltas based on errors from the
                # next layer.
                for i, neuron in enumerate(self.layer2.neurons):
                    # 2. Calculate errors of the neurons
                    error = sum([n.weights[i] * n.delta
                                 for n in self.layer3.neurons])
                    # 3. Calculate delta of the neurons
                    neuron.delta = error * sigmoid_derivative(neuron.output)

                # Hidden layer 1: Compute deltas based on errors from the
                # next layer.
                for i, neuron in enumerate(self.layer1.neurons):
                    # 2. Calculate errors of the neurons
                    error = sum([n.weights[i] * n.delta
                                 for n in self.layer2.neurons])
                    # 3. Calculate delta of the neurons
                    neuron.delta = error * sigmoid_derivative(neuron.output)

                # Update weights and biases based on computed deltas

                # Layer 3 (Output Layer)
                for neuron in self.layer3.neurons:
                    for j in range(len(neuron.weights)):
                        neuron.weights[j] += learning_rate * neuron.delta * self.layer2.neurons[j].output  # noqa
                    # 4. Apply learning rate when computing the biases
                    neuron.bias += learning_rate * neuron.delta

                # Layer 2 (Hidden Layer)
                for neuron in self.layer2.neurons:
                    for j in range(len(neuron.weights)):
                        neuron.weights[j] += learning_rate * neuron.delta * self.layer1.neurons[j].output  # noqa
                    # 4. Apply learning rate when computing the biases
                    neuron.bias += learning_rate * neuron.delta  # noqa

                # Layer 1 (Input Layer)
                for neuron in self.layer1.neurons:
                    for j in range(len(neuron.weights)):
                        neuron.weights[j] += learning_rate * neuron.delta * inputs[j]  # noqa
                    # 4. Apply learning rate when computing the biases
                    neuron.bias += learning_rate * neuron.delta  # noqa


if __name__ == '__main__':
    # Create a perceptron with 2 inputs, 6 neurons per hidden layer
    # and 1 output
    model = Perceptron(2, 6, 1)
    data = [[0, 0], [0, 1], [1, 0], [1, 1]]  # Inputs for the XOR problem
    labels = [[0], [1], [1], [0]]  # Target labels for the XOR problem
    model.fit(data, labels, 10000, 0.2)  # Training the perceptron

    # Test the trained perceptron on the XOR problem.
    for d in data:
        print(f'{d[0]} XOR {d[1]} => {round(model.forward(d)[0], 3)} => {round(model.forward(d)[0])}')  # noqa
