import sys
import yaml
from yaml import SafeLoader
from pathlib import Path
from constants import Constants
from utils.dataset import *
from utils.load_model import *
from utils.prediction import *

import warnings
warnings.filterwarnings("ignore")

# get predictions from the trained model for the images in the test set
def test(X, y, W1, b1, W2, b2, W3, b3):
    y_predicted = predict(X, W1, b1, W2, b2, W3, b3)
    test_accuracy = get_accuracy(y_predicted, y)
    return test_accuracy

if __name__ == "__main__":

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

    # load and preprocess the test set
    X_test, y_test = load_data(Constants.TEST_SET.value)
    X_test = preprocess_data(X_test)
    # load the trained model weights and biases
    W1, b1, W2, b2, W3, b3 = load_model_params(model_path)
    print("Model parameters loaded!")
    # get the accuracy of the trained model on the test set that contains images NOT seen by the model yet
    test_accuracy = test(X_test, y_test, W1, b1, W2, b2, W3, b3) * 100
    print("Test Accuracy: {:.2f}%".format(test_accuracy))