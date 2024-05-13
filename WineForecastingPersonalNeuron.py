import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ConceptPersonalBackward import Perceptron  # noqa

# Import dataset loading function
from sklearn.datasets import load_wine


# Set the seed
random.seed(10)

# Load the dataset
wine_ds = load_wine()
# Extract input values from the dataset
X = pd.DataFrame(wine_ds.data, columns=wine_ds.feature_names)[['flavanoids', 'proline', 'total_phenols']]  # noqa
# Extract output values from the dataset
y = pd.DataFrame(wine_ds.target, columns=['target'])

# Display the datasets
# X is our input values, they are used to predict target value
print('Wine features:')
print(X.head())
# y is a target value, that we want to predict; it has 3 target classes
print('Wine Classification Count:')
print(pd.DataFrame(y.value_counts()))

# Normalization
scalar = StandardScaler()
data = scalar.fit_transform(X.to_numpy())
print(f'Scaled data: \n{data[:3]}')

# OneHot Encoding
# Transform output labels into a Series with correct shape
labels = pd.Series(y.to_numpy().reshape(178))
# OneHot Encoding using pandas
labels = pd.get_dummies(labels).to_numpy(dtype='float32')
# Show lables format
print(f'\nOutput labels format:\n {labels[:3]}')

# Split data into train and test sets (40% of data will be used as test data)
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.4, random_state=21)  # noqa
# Show how much data we are training our model on
print(f'\nNumber of examples in train data: {data_train.shape[0]}')
print(f'Number of examples in test data: {data_test.shape[0]}')

# Model training

# Create a perceptron with 3 inputs, 10 neurons per hidden layer and 3 outputs
# input_size, hidden_size, output_size
model = Perceptron(3, 10, 3)

# Training the model with 100 epochs and learning rate 0.5
# training_data, labels, epochs, learning_rate
model.fit(data_train, labels_train, 100, 0.5)

# Model evaluation

# Get predictions of the model for every example in the test set
predictions = [model.forward(test_input) for test_input in data_test]

# Print the first 3 predicted labels and their true labels
for i in range(3):
    prediction = np.array(predictions[i])
    true_label = labels_test[i]
    print(f'True label: {true_label} => {np.argmax(true_label)}')
    print(f'Predicted: {prediction.round(3)} => {np.argmax(prediction)}')
    print()

# 3. Calculate accuracy of the model
accuracy = accuracy_score(
    [np.argmax(i) for i in labels_test],
    [np.argmax(prediction) for prediction in predictions]
)
print(f'Accuracy: {round(accuracy, 3)} => {100 * round(accuracy, 3)}%')
