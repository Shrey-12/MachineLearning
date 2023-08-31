import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target  # Multiclass labels

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def weightInitialization(n_features):
    w = np.zeros((n_features, 1))
    b = 0
    return w, b

def sigmoid_activation(result):
    final_result = 1 / (1 + np.exp(-result))
    return final_result

def model_optimize(w, b, X, Y):
    m = X.shape[0]

    # Prediction
    final_result = sigmoid_activation(np.dot(X, w) + b)
    cost = (-1/m) * np.sum(Y * np.log(final_result) + (1 - Y) * np.log(1 - final_result))

    # Gradient calculation
    dw = (1/m) * np.dot(X.T, (final_result - Y))
    db = (1/m) * np.sum(final_result - Y)

    grads = {"dw": dw, "db": db}

    return grads, cost

def model_predict(w, b, X, Y, learning_rate, no_iterations):
    costs = []
    for i in range(no_iterations):
        grads, cost = model_optimize(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        # Weight update
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

    coeff = {"w": w, "b": b}
    gradient = {"dw": dw, "db": db}

    return coeff, gradient, costs

def predict(final_pred):
    return np.argmax(final_pred, axis=1)

# Set hyperparameters
learning_rate = 0.01
no_iterations = 1500

# Initialize weights and bias for each class
num_classes = len(np.unique(y_train))
weights = []
biases = []

for _ in range(num_classes):
    w, b = weightInitialization(X_train.shape[1])
    weights.append(w)
    biases.append(b)

# Train a model for each class
coeffs = []
for i in range(num_classes):
    y_train_class = (y_train == i).astype(int)
    coeff, _, _ = model_predict(weights[i], biases[i], X_train, y_train_class, learning_rate, no_iterations)
    coeffs.append(coeff)

# Make predictions on the test set
final_preds = []
for i in range(num_classes):
    final_pred = sigmoid_activation(np.dot(X_test, coeffs[i]["w"]) + coeffs[i]["b"])
    final_preds.append(final_pred)

y_pred = predict(np.array(final_preds).T)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)