import os
# To remove anoying warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf  # noqa
import numpy as np  # noqa
import pandas as pd  # noqa


print('-' * 50)
##########################################################
# Basic Tensor Initializers
##########################################################
# tf.constant(): This is the simplest way to create a tensor.
# As the name suggests, tensors initialized with this method hold
# constant values and are immutable.
tensor_const = tf.constant([[1, 2], [3, 4]])
print('Tensor const:', tensor_const)

# tf.Variable(): Unlike tf.constant(), a tensor defined using
# tf.Variable() is mutable. This means its value can be changed,
# making it perfect for things like trainable parameters in models.
tensor_var = tf.Variable([[1, 2], [3, 4]])
print('Tensor var:', tensor_var)

# tf.zeros(): Create a tensor filled with zeros.
tensor_zeros = tf.zeros((3, 3))
print('Tensor zeros:', tensor_zeros)

# tf.ones(): Conversely, this creates a tensor filled with ones.
tensor_ones = tf.ones((2, 2))
print('Tensor ones:', tensor_ones)

# tf.fill(): Creates a tensor filled with a specific value.
tensor_fill = tf.fill((2, 2), 6)
print('Tensor filled:', tensor_fill)
# Tensor initialization

# tf.linspace() and tf.range(): These are fantastic for creating sequences.
# Generate a sequence of numbers starting from 0, ending at 9
tensor_range = tf.range(10)
print('Tensor range:', tensor_range)
# Create 5 equally spaced values between 0 and 10
tensor_linspace = tf.linspace(0, 10, 5)
print('Tensor linspace:', tensor_linspace)

# tf.random: Generates tensors with random values.
# Several distributions and functions are available within this module,
# like tf.random.normal() for values from a normal distribution,
# and tf.random.uniform() for values from a uniform distribution.
# Set random seed
tf.random.set_seed(72)
# Tensor of shape (2, 2) with random values normally distributed
# (default mean = 0, default std = 1)
tensor_random = tf.random.normal((2, 2), mean=4, stddev=0.5)
print('Tensor random normal:', tensor_random)
# Tensor of shape (2, 2) with random values uniformly distributed
# (default min = 0, default max = 1)
tensor_random = tf.random.uniform((2, 2), minval=-2, maxval=2)
print('Tensor random uniform:', tensor_random)

print('-' * 50)
##########################################################
# Converting Between Data Structures
##########################################################
# From Numpy Arrays: TensorFlow tensors and Numpy arrays are quite
# interoperable. Use tf.convert_to_tensor().
# Create a NumPy array based on a Python list
numpy_array = np.array([[1, 2], [3, 4]])
# Convert a NumPy array to a tensor
tensor_from_np = tf.convert_to_tensor(numpy_array)
print('Convert a NumPy array to a tensor:', tensor_from_np)

# From Pandas DataFrames: For those who are fans of data
# analysis with Pandas, converting a DataFrame or a Series
# to a TensorFlow tensor is straightforward.
# Use tf.convert_to_tensor() as well.
# Create a DataFrame based on dictionary
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
# Convert a DataFrame to a tensor
tensor_from_df = tf.convert_to_tensor(df.values)
print("Convert a DataFrame to a tensor:", tensor_from_df)

# Converting a constant tensor to a Variable: You can initialize a
# Variable using various tensor creation methods such as tf.ones(),
# tf.linspace(), tf.random, and so on. Simply pass the function or
# the pre-existing tensor to tf.Variable().
# Create a variable from a tensor
tensor = tf.random.normal((2, 3))
variable_1 = tf.Variable(tensor)
# Create a variable based on other generator
variable_2 = tf.Variable(tf.zeros((2, 2)))
# Display tensors
print('var1:', variable_1)
print('var2:', variable_2)

print('-' * 50)
