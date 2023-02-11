from utils.dataset import *
from utils.load_model import *
from utils.prediction import *

import warnings
warnings.filterwarnings("ignore")

################################# config variables #################################
test_set = 'test'
model_path = r"C:\Users\smsou\OneDrive\MY-DEVICE\Sync\Documents\GitHub\Neural-Network-from-Scratch\model\model.pkl" # add model_dir and model_file_path
####################################################################################

# get predictions from the trained model for the images in the test set
def test(X, y, W1, b1, W2, b2, W3, b3):
    y_predicted = predict(X, W1, b1, W2, b2, W3, b3)
    test_accuracy = get_accuracy(y_predicted, y)
    return test_accuracy

if __name__ == "__main__":

    # load and preprocess the test set
    X_test, y_test = load_data(test_set)
    X_test = preprocess_data(X_test)
    # load the trained model weights and biases
    W1, b1, W2, b2, W3, b3 = load_model_params(model_path)
    # get the accuracy of the trained model on the test set that contains images NOT seen by the model yet
    test_accuracy = test(X_test, y_test, W1, b1, W2, b2, W3, b3) * 100
    print("Test Accuracy: {:.2f}%".format(test_accuracy))