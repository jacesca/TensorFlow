import os
# To remove anoying warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf  # noqa


##########################################################
# Linear equations solution
##########################################################
# A system of linear equations can be represented in matrix
# form using the equation:
#       AX = B
# Where:
#       A is a matrix of coefficients.
#       X is a column matrix of variables.
#       B is a column matrix representing the values on the right
#         side of the equations.
# The solution to this system can be found using the formula:
#       X = A^-1 B
# Where A^-1 is the inverse of matrix A.

# Task - Objective
# Given a system of linear equations, use TensorFlow to solve it.
# You are given the following system of linear equations:
#       2x + 3y - z = 1
#       4x + y + 2z = 2
#       -x + 2y + 3z = 3
# (1) Represent the system of equations in matrix form (separate it
#     into matrices A and B).
# (2) Using TensorFlow, find the inverse of matrix A.
# (3) Multiply the inverse of matrix A by matrix B to find the solution
#     matrix X, which contains the values of x, y, and z.

print('-' * 50)
# 1. Define the matrices A and B
A = tf.constant([[2, 3, -1],
                 [4, 1, 2],
                 [-1, 2, 3]], dtype=tf.float64)
B = tf.constant([[1],
                 [2],
                 [3]], dtype=tf.float64)

# 2. Find the inverse of matrix A
A_inv = tf.linalg.inv(A)

# 3. Compute the solution matrix X (matrix multiplication)
X = tf.matmul(A_inv, B)

# Extract the values of x, y, and z
x, y, z = X[:, 0]

# Display the results
print('Linear equations solution')
print(f"x: {x:.2f}")
print(f"y: {y:.2f}")
print(f"z: {z:.2f}")
print('-' * 50)
