import numpy as np
from utils.network import *

# get predictions from the output layer
def get_predictions(A):
    return np.argmax(A, axis = 0)

# get accuracy value between 0 and 1
def get_accuracy(predictions, y):
    return np.sum(predictions == y) / y.size

# fit the neural network with optimzer: gradient descent
# alpha: learning rate
# iterations: epochs
def gradient_descent(X, y, alpha, iterations):
    # get initial values of the weights and biases
    W1, b1, W2, b2, W3, b3 = init_params(X)
    for i in range(iterations):
        # forward pass - apply the current weights, biases, and activation on the input data to get output labels
        Z1, A1, Z2, A2, Z3, A3 = forward(W1, b1, W2, b2, W3, b3, X)
        # backward pass - calculate loss and gradients with respect to weights and biases
        dW1, db1, dW2, db2, dW3, db3 = backward(W1, Z1, A1, W2, Z2, A2, W3, Z3, A3, X, y)
        # update weights and biases based on calculated gradients
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        # display prediction details every 10th epoch/iteration
        if i == 0 or (i+1) % 10 == 0:
            print("Iteration: {}".format(i + 1))
            predictions = get_predictions(A3)
            print("Predicted Labels: {}, True Labels: {}".format(predictions, y))
            print("Accuracy: {}".format(get_accuracy(predictions, y)))
    # return the final weights and biases
    return W1, b1, W2, b2, W3, b3