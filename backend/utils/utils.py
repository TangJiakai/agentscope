import json
import random
import subprocess

from loguru import logger


def try_serialize_dict(data):
    serialized_data = {}
    for key, value in data.items():
        try:
            json.dumps({key: value})
            serialized_data[key] = value
        except (TypeError, ValueError):
            pass
    return serialized_data

def run_sh_async(script_path: str, *args):
    command = ["bash", script_path, *args]
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logger.info(f"Run {script_path} with PID {process.pid}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_path}: {e}")

def run_sh_blocking(script_path: str, *args):
    command = ["bash", script_path, *args]
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logger.info(f"Run {script_path} with PID {process.pid}")

        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logger.error(f"Process returned non-zero exit status {process.returncode}")
            logger.error(stderr.decode())
        else:
            logger.info(stdout.decode())

    except Exception as e:
        logger.error(f"Error running {script_path}: {e}")

def traverse_gender(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() == "gender":
                return value.lower()
            if isinstance(value, (dict, object)) and not isinstance(value, (str, int, float, bool)):
                traverse_gender(value)
    elif hasattr(obj, '__dict__'):
        for key, value in obj.__dict__.items():
            if key.lower() == "gender":
                return value.lower()
            if isinstance(value, (dict, object)) and not isinstance(value, (str, int, float, bool)):
                traverse_gender(value)
    else:
        return random.choice(["female", "male"])
