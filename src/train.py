import sys
import yaml
from yaml import SafeLoader
from pathlib import Path
from constants import Constants
from utils.dataset import *
from utils.network import *
from utils.prediction import *
from utils.save_model import *

import warnings
warnings.filterwarnings("ignore")

# fit the neural network with the train set
# optimzer: gradient descent
# alpha: learning rate
# iterations: epochs considering batch size as 1
def gradient_descent(X, y, alpha, iterations):
    # get initial values of the weights and biases
    W1, b1, W2, b2, W3, b3 = init_params(X, units_layer_h1, units_layer_h2, units_layer_output)
    for i in range(iterations):
        # forward pass - apply the current weights, biases, and activation on the input data to get output labels
        Z1, A1, Z2, A2, Z3, A3 = forward(W1, b1, W2, b2, W3, b3, X)
        # backward pass - calculate loss and gradients with respect to weights and biases
        dW1, db1, dW2, db2, dW3, db3 = backward(W1, Z1, A1, W2, Z2, A2, W3, Z3, A3, X, y)
        # update weights and biases based on calculated gradients
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        # display prediction details every 10th iteration
        if i == 0 or (i+1) % intervals == 0:
            print("Iteration: {}".format(i + 1))
            predictions = get_predictions(A3)
            print("Predicted Labels: {}, True Labels: {}".format(predictions, y))
            train_accuracy = get_accuracy(predictions, y) * 100
            print("Train Accuracy: {:.2f}%".format(train_accuracy))
    # return the final weights and biases
    return W1, b1, W2, b2, W3, b3

if __name__ == '__main__':

    # get the desired parent directory as root path
    ROOT = Path(__file__).resolve().parents[1]

    # add ROOT to sys.path if not present
    if str(ROOT) not in sys.path:
        # add ROOT to sys.path
        sys.path.append(str(ROOT))

    # load the config and parse it into a dictionary
    with open(ROOT / Constants.CONFIG_PATH.value) as f:
        slice_config = yaml.load(f, Loader = SafeLoader)

    # get the model path from config
    model_path = ROOT / slice_config['model']['model_dir'] / slice_config['model']['model_name']
    # convert the path to a string in a format compliant with the current OS
    model_path = model_path.as_posix()

    # get the required variable values from config
    units_layer_h1 = slice_config['model']['units_layer_h1']
    units_layer_h2 = slice_config['model']['units_layer_h2']
    units_layer_output = slice_config['model']['units_layer_output']
    learning_rate = slice_config['optimizer']['learning_rate']
    iterations = slice_config['train']['iterations']
    intervals = slice_config['train']['intervals']

    # load and preprocess the train set
    X_train, y_train = load_data(Constants.TRAIN_SET.value)
    X_train = preprocess_data(X_train)
    # fit the defined neural network model with the training data and get the final weights and biases
    W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, y_train, learning_rate, iterations)
    # save the trained model parameters: weights and biases
    save_model_params(W1, b1, W2, b2, W3, b3, model_path)
    print("Model parameters saved!")