import numpy as np
from prop import forward_propagation

# Get NN layer sizes
def layer_size(X, n_hl, y):
    n_x = X.shape[0]
    n_y = y.shape[0]

    return (n_x, n_hl, n_y)

# Initialize W parameters
def initialize_parameters(n_x, n_hl, n_y):
    W1 = np.random.randn(n_hl, n_x) * 0.01
    b1 = np.zeros((n_hl, 1))

    W2 = np.random.randn(n_y, n_hl) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {'W1': W1,
                  'b1': b1,
                  'W2': W2,
                  'b2': b2}

    return parameters

# Gradient descent
def optimize(params, grads, learning_rate = 0.1):
    # Load in params
    W1 = params['W1']
    W2 = params['W2']
    b1 = params['b1']
    b2 = params['b2']

    # Load in grads
    dW1 = grads['dW1']
    dW2 = grads['dW2']
    db1 = grads['db1']
    db2 = grads['db2']

    # Update parameters
    W1 -= learning_rate * dW1
    W2 -= learning_rate * dW2
    db1 -= learning_rate * db1
    db2 = learning_rate * db2

    parameters = {'W1': W1,
                  'b1': b1,
                  'W2': W2,
                  'b2': b2}

    return parameters

# Make predictions
def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = np.argmax(A2, axis=0)
    return predictions

# Redefine y-labels
def reshape_labels(y):
    y_labels = y.reshape(-1)
    y_labels = np.eye(10)[y_labels]
    return y_labels
