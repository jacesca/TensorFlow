"""Essentially, a gradient is a set of partial derivatives."""
import os
# To remove anoying warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf  # noqa


##############################################################
# y = x^2
##############################################################
x = tf.Variable(3.0)

# Start recording the operations
with tf.GradientTape() as tape:
    # Define the calculations
    y = x * x

# Extract the gradient for the specific input (x)
grad = tape.gradient(y, x)
print('-' * 50)
print('y = x^2')
print(f'Result of y: {y}')
print(f'The gradient of y with respect to x is: {grad.numpy()}')
print(grad)
print(grad.numpy())
print('-' * 50)

##############################################################
# y = r_sum(x^2 + 2*z)
##############################################################
x = tf.Variable(tf.fill((2, 3), 3.0))
z = tf.Variable(5.0)

# Start recording the operations
with tf.GradientTape() as tape:
    # Define the calculations
    y = tf.reduce_sum(x * x + 2 * z)

# Extract the gradient for the specific inputs (x and z)
grad = tape.gradient(y, [x, z])

print('y = r_sum(x^2 + 2*z)')
print(f'Result of y: {y}')
print(f"The gradient of y with respect to x is:\n{grad[0].numpy()}")
print(f"The gradient of y with respect to z is: {grad[1].numpy()}")
print(grad)
print(grad[0])
print(grad[1])
print('-' * 50)

##############################################################
# y = x^2 + 2*x - 1
##############################################################
x = tf.Variable(2.0)

# Use Gradient Tape to record the computation of the function `f(x)`
with tf.GradientTape() as tape:
    y = x * x + 2 * x - 1

# Calculate the gradient of `f(x)` at the specified point
grad = tape.gradient(y, x)

# Print the results
print('y = x^2 + 2*x - 1')
print(f'Result of y: {y}')
print("The value of f(x) at x = 2 is:", y.numpy())
print("The derivative of f(x) at x = 2 is:", grad.numpy())
print('-' * 50)
