import os
# To remove anoying warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf  # noqa


##########################################################
# Arithmetic Operations
##########################################################
print('-' * 50)
# Addition
# For tensors addition we can use tf.add(), .assign_add()
# methods and a plus + sign. Also we can use broadcasting
# with the plus sign + or with the tf.add() method.
a = tf.Variable([1, 2, 3])
b = tf.constant([4, 5, 6])
# Perform element-wise addition with TF method
c1 = tf.add(a, b)
# Same as c1 calculation, but shorter
c2 = a + b
# Using broadcasting;
# Same as [1, 2, 3] + [3, 3, 3]
c3 = a + 3
# The most efficient one;
# Changes the object inplace without creating a new one;
# Result is the same as for c1 and c2.
a.assign_add(b)

print('ADITION')
print('TF method:\t', c1)
print('Plus sign:\t', c2)
print('Broadcasting:\t', c3)
print('Inplace change:\t', a)

print('-' * 50)
# Subtraction
# We have analogues of all methods for subtraction as for addition:
#   tf.add() changes into tf.subtract();
#   Plus sign + changes into minus sign -;
#   .assign_add() changes into .assign_sub()
a = tf.Variable([4, 5, 6])
b = tf.constant([1, 2, 3])
# Perform element-wise substraction
c1 = tf.subtract(a, b)
c2 = a - b
# Using broadcasting;
# Same as [4, 5, 6] - [3, 3, 3]
c3 = a - 3
# Inplace substraction
a.assign_sub(b)

print('SUBSTRACTION')
print('TF method:\t', c1)
print('Minus sign:\t', c2)
print('Broadcasting:\t', c3)
print('Inplace change:\t', a)

print('-' * 50)
# Multiplication (Element-wise)
# For multiplication, there isn't an inplace method since matrix
# multiplication inherently results in a new object. However,
# other operations have their counterparts:
#   tf.add() corresponds to tf.multiply();
#   The plus sign + corresponds to the asterisk sign *.
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
# Perform element-wise multiplication
c1 = tf.multiply(a, b)
c2 = a * b
# Using broadcasting;
# Same as [1, 2, 3] * [3, 3, 3]
c3 = a * 3

print('MULTIPLICATION')
print('TF method:\t', c1)
print('Asterisk sign:\t', c2)
print('Broadcasting:\t', c3)

print('-' * 50)
# Division
# Similar to multiplication, but with tf.divide() and / sign.
a = tf.constant([6, 8, 10])
b = tf.constant([2, 4, 5])
# Perform element-wise division
c1 = tf.divide(a, b)
c2 = a / b
# Using broadcasting;
# Same as [6, 8, 10] / [2, 2, 2]
c3 = a / 2

print('DIVISION')
print('TF method:\t', c1)
print('Asterisk sign:\t', c2)
print('Broadcasting:\t', c3)

print('-' * 50)
