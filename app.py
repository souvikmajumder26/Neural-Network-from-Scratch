import sys
import yaml
import logging
import subprocess
from pathlib import Path
from yaml import SafeLoader
from src.constants import Constants
from src.logger import get_logger

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    # get the desired parent directory as root path
    ROOT = Path(__file__).resolve().parents[0]

    # add ROOT to sys.path if not present
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    # load the config and parse it into a dictionary
    with open(ROOT / Constants.CONFIG_PATH.value) as f:
        slice_config = yaml.load(f, Loader = SafeLoader)

    # get the training script path from config
    train_path = ROOT / slice_config['train']['script_dir'] / slice_config['train']['script_name']
    # convert the path to a string in a format compliant with the current OS
    train_path = train_path.as_posix()

    # get the testing script path from config
    test_path = ROOT / slice_config['test']['script_dir'] / slice_config['test']['script_name']
    # convert the path to string in a format compliant with the current OS
    test_path = test_path.as_posix()

    # get the required variable values from config
    train_trigger = slice_config['train']['trigger']
    test_trigger = slice_config['test']['trigger']

    if train_trigger:
        # run the training script
        print("Subprocess 1: Running training script...")
        try:
            subprocess.run(['python', train_path], check = True)
            print("Training finished!")
        except Exception as e:
            print("Error running training script: %s" % str(e))

    if test_trigger:
        # run the testing script
        print("Subprocess 2: Running testing script...")
        try:
            subprocess.run(['python', test_path], check = True)
            print("Testing finished!")
        except Exception as e:
            print("Error running testing script: %s" % str(e))