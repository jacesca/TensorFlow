import os
# To remove anoying warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf  # noqa


# Define layer weights W, layer inputs I, and layer bias b
# one sample with 3 inputs
I = tf.constant([[0.5, 0.6, 0.1]], dtype=tf.float64)  # noqa
# 2 neurons with 3 inputs
W = tf.Variable([[0.5, 0.4, 0.7], [0.1, 0.9, 0.5]], dtype=tf.float64)
b = tf.Variable(0.1, dtype=tf.float64)

# Transpose weight matrix
W = tf.transpose(W)

# Compute the weighted sum
weighted_sum = tf.matmul(I, W)

# Add bias
biased_sum = weighted_sum + b

# Apply the sigmoid activation function
neuron_output = tf.sigmoid(biased_sum)

# Display results
print('-' * 50)
print('Transposed weight matrix:\n', W)
print('Weighted sum:\n', weighted_sum)
print('Biased sum:\n', biased_sum)
print('Output of the neuron:\n', neuron_output)
print('-' * 50)
