import sys
import yaml
import subprocess
from pathlib import Path
from yaml import SafeLoader

from src.constants import Constants
from src.logger import custom_logger

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

    logger.info("Pipeline starting...")

    if train_trigger:
        # run the training script
        logger.info("Subprocess 1: Running training script...")
        try:
            subprocess.run(['python', train_path], check = True)
            logger.info("Training finished!")
        except Exception as e:
            logger.error("Error running training script: %s" % str(e))

    if test_trigger:
        # run the testing script
        logger.info("Subprocess 2: Running testing script...")
        try:
            subprocess.run(['python', test_path], check = True)
            logger.info("Testing finished!")
        except Exception as e:
            logger.error("Error running testing script: %s" % str(e))
    
    logger.info("Pipeline completed!")