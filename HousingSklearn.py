import numpy as np

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import fetch_california_housing


# Hide warning: ConvergenceWarning: Stochastic Optimizer: Maximum iterations
# (20) reached and the optimization hasn't converged yet.
simplefilter("ignore", category=ConvergenceWarning)

# Loading the California housing dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Splitting the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scaling the features since neural networks are sensitive to feature scales
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Defining and training the MLPRegressor
mlp = MLPRegressor(
    hidden_layer_sizes=(10, 5),  # 2 layers, 1 of 10 neurons and 1 of 5 neurons
    max_iter=20,
    learning_rate_init=0.0001,
    random_state=42,
)

# Training the model
mlp.fit(X_train, y_train)

# Making predictions and evaluating the model
y_pred = mlp.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'MAE: {mae:.3f}')

# Setup a random distribution of hyperparameters
param_distributions = {
    'hidden_layer_sizes': [(n_neurons, n_neurons) for n_neurons in range(20, 31, 2)],  # noqa
    'learning_rate_init': [0.02, 0.01, 0.005],
    'max_iter': np.random.randint(10, 50, 10)
}

# Create the model
mlp = MLPRegressor()

# 4. Apply random search for 4 models (iterations)
random_search = RandomizedSearchCV(
    mlp,
    param_distributions,
    n_iter=4,
    cv=2,
    scoring='neg_mean_absolute_error',
    random_state=1
)
random_search.fit(X_train, y_train)

# Display the best parameters
print(f'Best parameters found: {random_search.best_params_}')

# Train the best model on the whole training data
best_mlp = random_search.best_estimator_
best_mlp.fit(X_train, y_train)

# 5. Evaluate the model
train_score = mean_absolute_error(y_train, best_mlp.predict(X_train))
test_score = mean_absolute_error(y_test, best_mlp.predict(X_test))

print(f'Train score: {train_score:.3f}')
print(f'Test score: {test_score:.3f}')
