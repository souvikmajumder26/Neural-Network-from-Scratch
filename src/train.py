from utils.dataset import *
from utils.network import *
from utils.prediction import *
from utils.save_model import *

import warnings
warnings.filterwarnings("ignore")

# fit the neural network with train set
# optimzer: gradient descent
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
            train_accuracy = get_accuracy(predictions, y) * 100
            print("Train Accuracy: {:.2f}%".format(train_accuracy))
    # return the final weights and biases
    return W1, b1, W2, b2, W3, b3

if __name__ == '__main__':

    # load and preprocess the train set
    X_train, y_train = load_data('train')
    X_train = preprocess_data(X_train)
    # fit the defined neural network model with the training data and get the final weights and biases
    W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, y_train, 0.10, 50)
    # save the trained model parameters: weights and biases
    model_path = r"C:\Users\smsou\OneDrive\MY-DEVICE\Sync\Documents\GitHub\Neural-Network-from-Scratch\model\model.pkl"
    save_model_params(W1, b1, W2, b2, W3, b3, model_path)
    print("Model parameters saved!")