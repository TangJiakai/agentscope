from utils.constants import *
import os


def check_load_adapter():
    if os.path.isfile(os.path.join(SAVE_DIR, ADAPTER_CONFIG_FILE_NAME)):
        return True
    return False


def check_dirs():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory {SAVE_DIR}")

    if not os.path.exists(LLM_DIR_PATH):
        raise FileNotFoundError(f"Please download the LLM model to {LLM_DIR_PATH}")

        