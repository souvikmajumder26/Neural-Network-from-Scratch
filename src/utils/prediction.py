import numpy as np
from utils.network import *

# get predictions from the output layer
def get_predictions(A):
    return np.argmax(A, axis = 0)

# get accuracy value between 0 and 1
def get_accuracy(predictions, y):
    return np.sum(predictions == y) / y.size

# get predictions from the trained model for given data
def predict(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    return predictions