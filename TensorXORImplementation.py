"""
Task
Create a neural network designed to predict XOR operation outcomes.
The network should consist of 2 input neurons, a hidden layer with 2
neurons, and 1 output neuron.

1.  Start by setting up the initial weights and biases. The weights should
    be initialized using a normal distribution, and biases should all be
    initialized to zero. Use the hyperparameters input_size, hidden_size,
    and output_size to define the appropriate shapes for these tensors.
2.  Utilize a function decorator to transform the train_step() function into
    a TensorFlow graph.
3.  Carry out forward propagation through both the hidden and output layers
    of the network. Use sigmoid activation function.
4.  Determine the gradients to understand how each weight and bias impacts
    the loss. Ensure the gradients are computed in the correct order,
    corresponding to the output variable names.
5.  Modify the weights and biases based on their respective gradients.
    Incorporate the learning_rate in this adjustment process to control the
    extent of each update.
"""
import os
# To remove anoying warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np  # noqa
import tensorflow as tf  # noqa


# Set seed
tf.random.set_seed(1)

# Set up input and output data
X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype='float32')
Y_data = np.array([[0], [1], [1], [0]], dtype='float32')

# Hyperparameters
input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.5

# Weights and biases for the hidden layer
W1 = tf.Variable(tf.random.normal((input_size, hidden_size)))
b1 = tf.Variable(tf.zeros((hidden_size)))

# Weights and biases for the output layer
W2 = tf.Variable(tf.random.normal((hidden_size, output_size)))
b2 = tf.Variable(tf.zeros((output_size)))


# Function that describes a single training step
@tf.function
def train_step(X, Y):
    with tf.GradientTape() as tape:
        # Forward pass for hidden layer
        z1 = tf.matmul(X, W1) + b1  # Calculate biased weighted sum
        a1 = tf.sigmoid(z1)  # Apply activation function

        # Forward pass for output layer
        z2 = tf.matmul(a1, W2) + b2  # Calculate biased weighted sum
        Y_pred = tf.sigmoid(z2)  # Apply activation function

        # Calculate loss (Mean Squared Error)
        loss = tf.reduce_mean((Y_pred - Y) ** 2)

    # Gradients
    dW1, db1, dW2, db2 = tape.gradient(loss, [W1, b1, W2, b2])

    # Update weights and biases
    W1.assign_sub(learning_rate * dW1)
    b1.assign_sub(learning_rate * db1)
    W2.assign_sub(learning_rate * dW2)
    b2.assign_sub(learning_rate * db2)

    return loss


# Number of epochs for the model to run through
epochs = 2500

# Training loop
for epoch in range(epochs):
    # Single training step
    loss = train_step(X_data, Y_data)
    # Display loss every 500 epochs
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
