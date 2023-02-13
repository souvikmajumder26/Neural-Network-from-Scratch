import sys
import yaml
from yaml import SafeLoader
from pathlib import Path

from utils.dataset import *
from utils.load_model import *
from utils.prediction import *

from constants import Constants
from logger import custom_logger

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

    # get the log file dir from config
    log_dir = ROOT / slice_config['log']['log_dir']
    # make the directory if it does not exist
    log_dir.mkdir(parents = True, exist_ok = True)
    # get the log file path
    log_path = log_dir / slice_config['log']['log_name']
    # convert the path to string in a format compliant with the current OS
    log_path = log_path.as_posix()

    # get the value for the log level from config
    log_level = slice_config['log']['log_level']

    logger = custom_logger("Neural Network Logs", log_path, log_level)

    # get the model path from config
    model_path = ROOT / slice_config['model']['model_dir'] / slice_config['model']['model_name']
    # check if the model path exists
    if not model_path.exists():
        logger.error("Model does not exist at %s!" % model_path)
    # convert the path to a string in a format compliant with the current OS
    model_path = model_path.as_posix()

    # load and preprocess the test set
    X_test, y_test = load_data(Constants.TEST_SET.value)
    X_test = preprocess_data(X_test)
    # load the trained model weights and biases
    W1, b1, W2, b2, W3, b3 = load_model_params(model_path)
    logger.info("Model parameters loaded!")
    # get the accuracy of the trained model on the test set that contains images NOT seen by the model yet
    test_accuracy = test(X_test, y_test, W1, b1, W2, b2, W3, b3) * 100
    logger.info("Test Accuracy: {:.2f}%".format(test_accuracy))