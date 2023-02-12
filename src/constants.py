from pathlib import Path
from enum import Enum

class Constants(Enum):
    VERSION = "C_0"
    USE_CASE = "Neural_Network_from_Scratch"
    LOGS_DIR = "logs"
    CONFIG_PATH = Path("config/config.yaml")
    TRAIN_SET = "train"
    TEST_SET = "test"