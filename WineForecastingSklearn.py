import pandas as pd
import random

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import dataset loading function
from sklearn.datasets import load_wine


# Set the seed
random.seed(10)

# Load the dataset
wine_ds = load_wine()
# Extract input values from the dataset
X = pd.DataFrame(wine_ds.data, columns=wine_ds.feature_names)[['flavanoids', 'proline', 'total_phenols']]  # noqa
# X = pd.DataFrame(wine_ds.data, columns=wine_ds.feature_names)
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
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=21)  # noqa
# Show how much data we are training our model on
print(f'\nNumber of examples in train data: {X_train.shape[0]}')
print(f'Number of examples in test data: {X_test.shape[0]}')


# Set up parameters of the model
model = MLPClassifier(
    max_iter=100,
    hidden_layer_sizes=(10, 10),
    learning_rate_init=0.5,
    random_state=21
)

# Train the model
model.fit(X_train, y_train)

# 3. Evaluate the model
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score, 3)} => {100 * round(score, 3)}%')
