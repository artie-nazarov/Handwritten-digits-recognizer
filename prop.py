import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Softmax function
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)

# Compute Cross-Entropy Cost
def cross_entropy(A2, y):
    m = y.shape[1]
    loss = y * np.log(A2) + (1-y) * np.log(1-A2)
    cost = -(1/m) * np.sum(loss)

    return cost

# Forward propagation
def forward_propagation(X, params):
    # Load in parameters
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    # Forward pass
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    cache = {'Z1': Z1,
             'A1': A1,
             'Z2': Z2,
             'A2': A2}

    return A2, cache

# Backward Propagation
def backward_propagation(X, y, parameters, cache):
    m = y.shape[1]

    # Load in params
    W1 = parameters['W1']
    W2 = parameters['W2']

    # Load in cache
    A1 = cache['A1']
    A2 = cache['A2']

    # Backward pass
    dZ2 = A2 - y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {'dW2': dW2,
             'dW1': dW1,
             'db2': db2,
             'db1': db1}

    return grads

