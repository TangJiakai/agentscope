from llmtuning.code.utils.constants import *
import os


def check_load_adapter():
    if os.path.isfile(os.path.join(SAVE_DIR, ADAPTER_CONFIG_FILE_NAME)):
        return True
    return False