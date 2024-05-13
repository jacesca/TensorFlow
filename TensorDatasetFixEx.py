"""
Objective
Prepare a corrected and complete dataset for weather prediction
(1) Data Correction:
    You found out that the readings on the 2nd and 5th rows of the
    main_dataset were inaccurate. Replace these rows in the main_dataset
    with the rows from error_correction_data.
(2) Incorporate Additional Data:
    Concatenate the main_dataset with additional_data to incorporate the
    new readings.
(3) Batch Reshaping:
    For the purpose of batch training, you want to divide the dataset into
    batches with 3 readings per batch. Reshape complete_dataset, where the
    first dimension represents the batch size, and the second dimension
    represents the number of readings per batch.
"""
import os
# To remove anoying warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf  # noqa


# Main dataset with six readings
main_dataset = tf.Variable([
    [25.0, 1012.5, 0.4, 0.6],    # Sample 1
    [22.0, 1010.2, 0.5, 0.7],    # Erroneous reading
    [24.5, 1011.5, 0.42, 0.62],  # Sample 3
    [26.0, 1012.8, 0.43, 0.65],  # Sample 4
    [23.0, 1010.0, 0.44, 0.66],  # Erroneous reading
    [24.0, 1012.3, 0.45, 0.68]   # Sample 6
])

# Correction data for the main dataset
error_correction_data = tf.constant([
    [22.5, 1010.5, 0.51, 0.71],  # Corrected reading for the 2nd sample
    [23.5, 1010.3, 0.46, 0.67]   # Corrected reading for the 5th sample
])

# Additional data readings
additional_data = tf.constant([
    [24.2, 1012.6, 0.47, 0.69],  # Sample 7
    [25.5, 1013.0, 0.48, 0.7],   # Sample 8
    [24.8, 1012.9, 0.49, 0.72]   # Sample 9
])

# Solution

# Replace the 2nd and 5th rows in the main_dataset with error_correction_data
main_dataset[1].assign(error_correction_data[0])
main_dataset[4].assign(error_correction_data[1])

# Concatenate main_dataset and additional_data
complete_dataset = tf.concat([main_dataset, additional_data], axis=0)

# Reshape the complete_dataset into 3 batches of 3 samples each
batched_dataset = tf.reshape(complete_dataset, (3, 3, 4))

# Output the final batched dataset
print(batched_dataset)
