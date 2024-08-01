import json
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


def run_sh(script_path: str, *args):
    command = ["bash", script_path, *args]
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logger.info(f"Run {script_path} with PID {process.pid}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_path}: {e}")
