# Tensor transformations are crucial when working with data.
# As data science tasks, you'll find that the data you deal with
# isn't always in the format you need. Here will introduce you to
# methods in TensorFlow that allow you to manipulate the structure
# and content of tensors to fit your needs.
import os
# To remove anoying warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf  # noqa


##########################################################
# Reshaping Tensors
##########################################################
print('-' * 50)
print('Reshaping Tensors')
tensor = tf.constant([[1, 2], [3, 4], [5, 6]])
print('Original Tensor:', tensor)
# Reshape the tensor to shape (2, 3)
reshaped_tensor = tf.reshape(tensor, (2, 3))
print('Reshaped Tensor (2,3):', reshaped_tensor)
# Reshape the tensor to shape (6, 1);
# The size of the first dimention is determined automatically
reshaped_tensor = tf.reshape(tensor, (-1, 1))
print('Reshaped Tensor (6,1):', reshaped_tensor)


##########################################################
# Slicing
##########################################################
print('-' * 50)
print('Slicing')
tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print('Original Tensor:', tensor)
# Slice tensor to extract sub-tensor from index (0, 1) of size (1, 2)
sliced_tensor = tf.slice(tensor, begin=(0, 1), size=(1, 2))
print('Sliced Tensor:', sliced_tensor)
# Slice tensor to extract sub-tensor from index (1, 0) of size (2, 2)
sliced_tensor = tf.slice(tensor, (1, 0), (2, 2))
print('Sliced Tensor:', sliced_tensor)

##########################################################
# Modifying Data
##########################################################
print('-' * 50)
print('Modifying Data')
tensor = tf.Variable([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print('Original Tensor:', tensor)

# Change the entire first row
tensor[0, :].assign([0, 0, 0])
print('Modified Tensor:', tensor)

# Modify the second and the third columns
tensor[:, 1:3].assign(tf.fill((3, 2), 1))
print('Modified Tensor:', tensor)

##########################################################
# Concatenating
##########################################################
print('-' * 50)
print('Concatenating')
tensor1 = tf.constant([[1, 2, 3], [4, 5, 6]])
tensor2 = tf.constant([[7, 8, 9]])
print('Original Tensor 1:', tensor1)
print('Original Tensor 2:', tensor2)
# Concatenate tensors vertically (along rows)
concatenated_tensor = tf.concat([tensor1, tensor2], axis=0)
print('Concatenated Tensor:', concatenated_tensor)
# Create another set of tensors
tensor3 = tf.constant([[1, 2], [3, 4]])
tensor4 = tf.constant([[5], [6]])
print('Original Tensor 3:', tensor3)
print('Original Tensor 4:', tensor4)
# Concatenate tensors horizontally (along columns)
concatenated_tensor = tf.concat([tensor3, tensor4], axis=1)
print('Concatenated Tensor:', concatenated_tensor)

print('-' * 50)
