import json
import random
import re
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


def run_sh_train_blocking(script_path: str, *args):
    command = ["stdbuf", "-oL", "bash", script_path, *args]

    # 定义正则表达式来匹配 "Training progress: <number>"
    progress_pattern = re.compile(r"Training progress:\s*(\d+\.\d+)")

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        logger.info(f"Run {script_path} with PID {process.pid}")

        # 实时获取输出
        while True:
            # 逐行读取标准输出
            stdout_line = process.stdout.readline()

            if stdout_line:
                # 尝试匹配并提取 "Training progress" 后的数值
                progress_match = progress_pattern.search(stdout_line)
                if progress_match:
                    progress = float(progress_match.group(1))  # 提取并转换为浮点数
                    import backend.app as app

                    app.train_progress = progress
                    logger.info(f"Current training progress: {progress}")
                else:
                    logger.info(stdout_line.strip())  # 打印其他标准输出内容

            # 检查进程是否已经结束
            if process.poll() is not None and not stdout_line:
                break

        # 获取剩余输出（如果有）
        stdout, stderr = process.communicate()

        if stdout:
            logger.info(stdout)
        if stderr:
            logger.error(stderr)

        if process.returncode != 0:
            logger.error(f"Process returned non-zero exit status {process.returncode}")
        else:
            logger.info("Process completed successfully.")

    except Exception as e:
        logger.error(f"Error running {script_path}: {e}")


def traverse_gender(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() == "gender":
                return value.lower()
            if isinstance(value, (dict, object)) and not isinstance(
                value, (str, int, float, bool)
            ):
                traverse_gender(value)
    elif hasattr(obj, "__dict__"):
        for key, value in obj.__dict__.items():
            if key.lower() == "gender":
                return value.lower()
            if isinstance(value, (dict, object)) and not isinstance(
                value, (str, int, float, bool)
            ):
                traverse_gender(value)
    else:
        return random.choice(["female", "male"])


if __name__ == "__main__":
    run_sh_train_blocking(
        "/data/gaoheyang/new_GeneralSimulation/exp2/scripts/tune_llm.sh", "sft"
    )
