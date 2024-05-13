import os
# To remove anoying warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf  # noqa
import numpy as np  # noqa
import pandas as pd  # noqa


print('-' * 50)
##########################################################
# Available Data Types in TensorFlow
##########################################################
# tf.float16, tf.float32, tf.float64
# tf.int8, tf.int16, tf.int32, tf.int64
# tf.uint8, tf.uint16, tf.uint32, tf.uint64
# tf.bool
# tf.string

# Creating a tensor of type float16
tensor_float = tf.constant([-1.2, 2.3, 3.4], dtype=tf.float16)
# Creating a tensor of type int64
tensor_int = tf.constant([1, 2, 3], dtype=tf.int64)

# Display tensors
print('Tensor float:', tensor_float)
print('Tensor int:', tensor_int)

# Converting Between Data Types
# Convert our tensor_float from float32 to int32
tensor_int_converted = tf.cast(tensor_float, dtype=tf.int32)
# Display a tensor
print('Tensor converted to int:', tensor_int_converted)

print('-' * 50)
##########################################################
# Arithmetic Operations
##########################################################
