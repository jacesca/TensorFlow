"""
Function Decorator
A Function Decorator is a tool that 'wraps' around a function to modify its
behavior. In TensorFlow, the most commonly used decorator is @tf.function,
which converts a Python function into a TensorFlow graph.
"""
import os
# To remove anoying warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time  # noqa
import tensorflow as tf  # noqa


##########################################################
# @tf.function applied to a basic function
##########################################################
def compute_area(radius):
    return 3.1415 * radius ** 2


@tf.function
def optimized_compute_area(radius):
    return 3.1415 * radius ** 2


# Run once for the graph to set up
area = optimized_compute_area(tf.constant(3.0))

# Measure execution time for basic function
start = time.time()
area = compute_area(tf.constant(3.0))
duration = time.time() - start
print('Simple function Time: ', duration)

# Measure execution time for optimized function
start = time.time()
area = optimized_compute_area(tf.constant(3.0))
duration = time.time() - start
print('Tensor function Time: ', duration)

# print(f"The area is: {area.numpy()}")
# print(f"The area is: {area}")
print('-' * 50)


##########################################################
# Example with Gradient Tape
##########################################################
def compute_gradient(x):
    with tf.GradientTape() as tape:
        y = x * x * x
    return tape.gradient(y, x)


@tf.function
def optimized_compute_gradient(x):
    with tf.GradientTape() as tape:
        y = x * x * x
    return tape.gradient(y, x)


# Prepare the variables
x = tf.Variable(3.0)

# Run once for the graph to set up
grad = optimized_compute_gradient(x)

# Measure execution time for basic function
start = time.time()
grad = compute_gradient(x)
duration = time.time() - start
print('Gradient function Time: ', duration)

start = time.time()
grad = optimized_compute_gradient(x)
duration = time.time() - start
print('Tensor function Time: ', duration)

# print(f"The gradient at x = {x.numpy()} is {grad.numpy()}")
print('-' * 50)


##########################################################
# Example with Conditional Logic
##########################################################
def compute_gradient_conditional(x):
    with tf.GradientTape() as tape:
        if tf.reduce_sum(x) > 0:
            y = x * x
        else:
            y = x * x * x
    return tape.gradient(y, x)


@tf.function
def optimized_compute_gradient_conditional(x):
    with tf.GradientTape() as tape:
        if tf.reduce_sum(x) > 0:
            y = x * x
        else:
            y = x * x * x
    return tape.gradient(y, x)


# Prepare the variables
x = tf.Variable([-2.0, 2.0])

# Run once for the graph to set up
grad = optimized_compute_gradient_conditional(x)

start = time.time()
grad = compute_gradient_conditional(x)
duration = time.time() - start
print('Conditional function Time: ', duration)

start = time.time()
grad = optimized_compute_gradient_conditional(x)
duration = time.time() - start
print('Tensor function Time: ', duration)

# print(f"The gradient at x = {x.numpy()} is {grad.numpy()}")
print('-' * 50)


##########################################################
# Comparing the execution times of two TensorFlow functions
##########################################################
def matrix_multiply_basic(mat1, mat2):
    x = tf.matmul(mat1, mat2)
    for i in range(100):
        x = tf.matmul(x, mat2)
    return tf.reduce_mean(x)


@tf.function
def matrix_multiply_optimized(mat1, mat2):
    x = tf.matmul(mat1, mat2)
    for i in range(100):
        x = tf.matmul(x, mat2)
    return tf.reduce_mean(x)


# Create random matrices
mat1 = tf.random.uniform((100, 100), minval=0., maxval=0.01)
mat2 = tf.random.uniform((100, 100), minval=0., maxval=0.01)

# Run once for the graph to set up
result_optimized = matrix_multiply_optimized(mat1, mat2)

# Measure execution time for basic function
start_time = time.time()
result_basic = matrix_multiply_basic(mat1, mat2)
basic_time = time.time() - start_time

# Measure execution time for optimized function
start_time = time.time()
result_optimized = matrix_multiply_optimized(mat1, mat2)
optimized_time = time.time() - start_time

# Print results
print(f"Time taken by basic function: {basic_time:.4f} seconds")
print(f"Time taken by optimized function: {optimized_time:.4f} seconds")
print('-' * 50)
