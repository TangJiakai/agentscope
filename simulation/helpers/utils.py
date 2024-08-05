import threading
from functools import wraps
import yaml
import os
import json

from agentscope.file_manager import file_manager, _DEFAULT_CFG_NAME
from agentscope.message import Msg, serialize, deserialize

from simulation.memory import *


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    

def save_configs(configs):
    with open(
        os.path.join(file_manager.dir_root, _DEFAULT_CFG_NAME),
        "r+",
        encoding="utf-8",
    ) as file:
        cfg = json.load(file)
        cfg.update(configs)
        file.seek(0)
        json.dump(cfg, file, indent=4)
        file.truncate()
    

def setup_memory(memory_config):
    memory = eval(memory_config["class"])(**memory_config["args"])
    return memory


def rpc_client_post(agent_client, fun=None, params=None, msg=None):
    return deserialize(agent_client.call_agent_func(
        func_name="_reply",
        value=serialize(
            Msg("assistant", msg, role="assistant", fun=fun, params=params)
        )
    ))


def rpc_client_get(agent_client, msg):
    return deserialize(agent_client.update_placeholder(msg["task_id"]))


def rpc_client_post_and_get(agent_client, fun=None, params=None, msg=None):
    return rpc_client_get(
        agent_client,
        rpc_client_post(agent_client, fun=fun, params=params, msg=msg)
    )


def get_assistant_msg(content=None, **kwargs):
    return Msg("assistant", content, role="assistant", **kwargs)