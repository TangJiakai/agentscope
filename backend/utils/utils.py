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
        result = subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logger.info(f"Run {script_path} output:\n{result.stdout.decode()}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Run {script_path} failed with return code: {e.returncode}")
        logger.error(f"Run {script_path} error output:\n{e.stderr.decode()}")
