import os
# To remove anoying warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf  # noqa


##########################################################
# Linear Algebra Operations
##########################################################
# Matrix Multiplication
# There are two equivalent approaches for matrix multiplication:
# The tf.matmul() function.
# Using the @ operator.
# Which of the following is an alternative to matrix multiplication
# using tf.matmul?
#       tf.tensordot with axes=1 parameter
print('-' * 50)
matrix1 = tf.constant([[1, 2], [3, 4], [2, 1]])
matrix2 = tf.constant([[2, 0, 2, 5], [2, 2, 1, 3]])
# Multiply the matrices
product1 = tf.matmul(matrix1, matrix2)
product2 = matrix1 @ matrix2
# Display tensors
print('Matrix Multiplication')
print(product1)
print(product2)

# Matrix Inversion
# You can obtain the inverse of a matrix using the tf.linalg.inv()
# function. Additionally, let's verify a fundamental property of
# the inverse matrix.
print('-' * 50)
print('Matrix Inversion')
matrix = tf.constant([[1., 2.], [3., 4.]])
# Compute the inverse of a matrix
inverse_mat = tf.linalg.inv(matrix)
# Check the result
identity = matrix @ inverse_mat
# Display tensors
print(inverse_mat)
print(identity)

# Transpose
# You can obtain a transposed matrix using the tf.transpose() function.
print('-' * 50)
matrix = tf.constant([[1, 2], [3, 4], [2, 1]])
# Get the transpose of a matrix
transposed = tf.transpose(matrix)
# Display tensors
print('Transpose')
print(matrix)
print(transposed)

# Dot Product
# You can obtain a dot product using the tf.tensordot() function.
# By setting up an axes argument you can choose along which axes to
# calculate a dot product. E.g. for two vectors by setting up axes=1
# you will get the classic dot product between vectors. But when
# setting axes=0 you will get broadcasted matrix along 0 axes:
# In the dot product >> The number of columns in the first matrix must
#                       equal the number of rows in the second matrix.
print('-' * 50)
matrix1 = tf.constant([1, 2, 3, 4])
matrix2 = tf.constant([2, 0, 2, 5])
# Compute the dot product of two tensors
dot_product_axes1 = tf.tensordot(matrix1, matrix2, axes=1)
dot_product_axes0 = tf.tensordot(matrix1, matrix2, axes=0)
# [[ 2  0  2  5]   >> 1*2, 1*0, 1*2, 1*5
#  [ 4  0  4 10]   >> 2*2, 2*0, 2*2, 2*5
#  [ 6  0  6 15]   >> 3*2, 3*0, 3*2, 3*5
#  [ 8  0  8 20]]  >> 4*2, 4*0, 4*2, 4*5

# Display tensors
print('Dot Product')
print(dot_product_axes1)
print(dot_product_axes0)

print('-' * 50)
