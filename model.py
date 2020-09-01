import numpy as np
from suplement import layer_size, initialize_parameters, optimize
from prop import cross_entropy, forward_propagation, backward_propagation

# Neural Network Model
def model(X, y, n_hl, num_iterations, print_cost = False):

    # Define useful values
    m = y.shape[1]

    # Get layer sizes
    n_x, _, n_y = layer_size(X, n_hl, y)

    # Initialize weights
    parameters = initialize_parameters(n_x, n_hl, n_y)
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    # Loop over number of iterations
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)

        cost = cross_entropy(A2, y)

        grads = backward_propagation(X, y, parameters, cache)

        parameters = optimize(parameters, grads)

        if (i % 100 == 0 and print_cost):
            print('Cost after %i iterations: %f' % (i, cost))

    return parameters
