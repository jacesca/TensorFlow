import os
# To remove anoying warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# # To remove the message:
# # I tensorflow/core/util/port.cc:113] oneDNN custom operations are on.
# # You may see slightly different numerical results due to floating-point
# # round-off errors from different computation orders. To turn them off,
# # set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf  # noqa


print('Version:', tf.__version__)

# Create tensors
# 2D tensors: Table Data, Text Sequences, Numerical Sequences
#             Example:
#             - You have a table of patient records with 500
#               patients. Each record has 8 features like age,
#               blood type, height, and weight.
#               (500, 8)
#             - A novel is processed word by word, and it has
#               1000 words in total. If each word is represented
#               using embeddings of size 20
#               (1000, 20)
#             - An environmental monitoring system captures data
#               of 4 different metrics (like CO2 level, temperature,
#               humidity, and air pressure) over 12 hours.
#               (360, 4)
# 3D tensors: Image Processing
# 4D tensors: Video Processing
#             - You have a dataset of 200 grayscale images for a
#               machine learning project. Each image is 128x128
#               pixels. Grayscale images only have 1 channel.
#               (200, 128, 128, 1)
tensor_0D = tf.constant(5)  # 0-dimensional tensor
tensor_1D = tf.constant([1, 2, 3])
tensor_2D = tf.constant([[1, 2], [3, 4], [5, 6]])
tensor_3D = tf.constant([[[6, 9, 6], [1, 1, 2], [9, 7, 3]],
                         [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                         [[5, 6, 3], [5, 3, 5], [8, 8, 2]]])

# Display tensors
print(f"""
{'-' * 50}
{tensor_0D}
ndim: {tensor_0D.ndim}
shape: {tensor_0D.shape}
type: {tensor_0D.dtype}
{'-' * 50}
{tensor_1D}
ndim: {tensor_1D.ndim}
shape: {tensor_1D.shape}
type: {tensor_1D.dtype}
{'-' * 50}
{tensor_2D}
ndim: {tensor_2D.ndim}
shape: {tensor_2D.shape}
type: {tensor_2D.dtype}
{'-' * 50}
{tensor_3D}
ndim: {tensor_3D.ndim}
shape: {tensor_3D.shape}
type: {tensor_3D.dtype}
{'-' * 50}
""")

# Batches
# Examples:
# (1) Let's say we have 2048 data samples, each with a shape of (base shape).
#     This gives us a tensor of (2048, base shape). If we break this data into
#     batches of 32 samples, we'll end up with 64 batches, as 64 * 32 = 2048.
#     Shape: (64, 32, base shape).
# (2) A surveillance system records videos in batches for processing.
#     If you have batches of 10 videos, each 5 minutes long, with a frame
#     captured every second and each frame is a 512x512 pixel colored image.
#     Shape: (10, 300, 512, 512, 3)
# (3) A website tracks user activities for 50 users over a month (30 days).
#     Every day, it logs 10 different metrics (like time spent on site, clicks,
#     purchases, etc.) for each user (15000 values in total). What tensor shape
#     encapsulates this data?
#     Shape: (30, 50, 10)
