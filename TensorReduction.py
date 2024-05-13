import os
# To remove anoying warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf  # noqa


##########################################################
# Reduction Operations
##########################################################
# Sum, Mean, Maximum, and Minimum
# TensorFlow offers the following methods for these calculations:
#   tf.reduce_sum(): Computes the sum of all elements in the tensor or
#                    along a specific axis.
#   tf.reduce_mean(): Calculates the mean of the tensor elements.
#   tf.reduce_max(): Determines the maximum value in the tensor.
#   tf.reduce_min(): Finds the minimum value in the tensor.
print('-' * 50)
print('Reduction Operations')
tensor = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
print('Original Tensor:', tensor)
# Calculate sum of all elements
total_sum = tf.reduce_sum(tensor)
print("Reduce Sum:", total_sum.numpy())
# Calculate mean of all elements
mean_val = tf.reduce_mean(tensor)
print("Reduce Mean:", mean_val.numpy())
# Determine the maximum value
max_val = tf.reduce_max(tensor)
print("Reduce Maximum:", max_val.numpy())
# Find the minimum value
min_val = tf.reduce_min(tensor)
print("Reduce Minimum:", min_val.numpy())

##########################################################
# Operations along specific axes
##########################################################
# Tensors can have multiple dimensions, and sometimes we want to
# perform reductions along a specific axis. The axis parameter
# allows us to specify which axis or axes we want to reduce.
#   > axis=0: Perform the operation along the rows (resulting in a column
#             vector).
#   > axis=1: Perform the operation along the columns (resulting in a row
#             vector).
#   > It's possible to reduce along multiple axes simultaneously by providing
#     a list to the axis parameter.
#   > When the tensor's rank is reduced, you can use keepdims=True to retain
#     the reduced dimension as 1.
print('-' * 50)
print('Operations along specific axes')
tensor = tf.constant([[1., 2.], [3., 4.], [5., 6.]])
print('Original Tensor:', tensor)
# Calculate the sum of each column
col_sum = tf.reduce_sum(tensor, axis=0)
print("Column-wise Sum:", col_sum.numpy())
# Calculate the maximum of each row
col_max = tf.reduce_max(tensor, axis=1)
print("Row-wise Max:", col_max.numpy())
# Calculate the mean of the whole tensor (reduce along both directions)
# Equivalent to not specifying the axis at all
total_mean = tf.reduce_mean(tensor, axis=(0, 1))
print("Total Mean:", total_mean.numpy())
# Calculate the mean of the whole tensor (keeping reduced dimensions)
total_mean_dim = tf.reduce_mean(tensor, axis=(0, 1), keepdims=True)
print("Total Mean (saving dimensions):", total_mean_dim.numpy())

# When you execute a reduction operation along a specific axis, you
# essentially eliminate that axis from the tensor, aggregating all the
# tensors within that axis element-wise. The same effect will remain
# for any number of dimensions.
print()
tensor = tf.constant([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
    [[9, 10], [11, 12]]
])
print('Original Tensor:', tensor)
# Calculate the sum along axis 0
sum_0 = tf.reduce_sum(tensor, axis=0)
print("Sum axis 0 (col):\n", sum_0.numpy())
#  [[15 18] >> 1+5+9, 2+6+10
#   [21 24]] >> 3+7+11, 4+8+12
# Calculate the sum along axis 1
sum_1 = tf.reduce_sum(tensor, axis=1)
print("Sum axis 1 (row):\n", sum_1.numpy())
#  [[ 4  6]   >> 1+3, 2-4
#  [12 14]    >> 5+7, 6+8
#  [20 22]]   >> 9+11, 10+12
# Calculate the sum along axes 0 and 1
sum_0_1 = tf.reduce_sum(tensor, axis=(0, 1))
print("Sum axes 0 and 1:\n", sum_0_1.numpy())
#  [36 42] 1+5+9+3+7+11, 2+6+10+4+8+12

print('-' * 50)
