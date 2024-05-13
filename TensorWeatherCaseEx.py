"""
You are a data scientist at a weather research agency.
You've been given a tensor containing weather readings from various cities
over several days. The tensor has the following structure:
    > Dimension 1: Represents different cities.
    > Dimension 2: Represents different days.
    > Each entry in the tensor is a tuple of (temperature, humidity).

Objective
    > Calculate the average temperature for each city over all the days.
    > Calculate the maximum humidity reading across all cities for each day.
"""
import os
# To remove anoying warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf  # noqa


# Sample weather data tensor
weather_data = tf.constant([
    [[20, 60], [21, 58], [19, 62]],      # City 1 readings
    [[22, 55], [23, 54], [22, 56]],      # City 2 readings
    [[18, 65], [19, 67], [18, 64]]       # City 3 readings
], dtype=tf.float32)

# 1. Calculate the average temperature for each city over all the days
avg_temperature_per_city = tf.reduce_mean(weather_data[:, :, 0], axis=1)
print("Average temperature for each city:", avg_temperature_per_city.numpy())

# 2. Calculate the maximum humidity reading across all cities for each day.
max_humidity_per_day = tf.reduce_max(weather_data[:, :, 1], axis=0)
print("Maximum humidity for each day:", max_humidity_per_day.numpy())
