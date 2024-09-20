import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import argparse
import math
import json

from simulation.helpers.utils import load_json
from simulation.helpers.constants import *

scene_path = os.path.dirname(os.path.abspath(__file__))
AGENT_CONFIG = "RecUserAgent_configs.json"

def parse_args() -> argparse.Namespace:
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--base_port", type=int, default=13000)
    parser.add_argument("--server_num_per_host", type=int, default=1)
    return parser.parse_args()


def save_agent_configs(agent_configs, file_path):
    with open(file_path, "w", encoding='utf-8') as file:
        json.dump(agent_configs, file, ensure_ascii=False, indent=4)


def main(args):
    host = args.host
    base_port = args.base_port
    server_num_per_host = args.server_num_per_host
    available_port_num = server_num_per_host

    agent_configs = load_json(os.path.join(scene_path, CONFIG_DIR, AGENT_CONFIG))

    print("len(agent_configs):", len(agent_configs))

    total_agent_num = len(agent_configs)
    agent_num_per_server = math.ceil(total_agent_num / available_port_num)
    print("agent_num_per_server:", agent_num_per_server)

    for i, agent_config in enumerate(agent_configs):
        agent_config["args"]["host"] = host
        agent_config["args"]["port"] = base_port + i % available_port_num

    save_agent_configs(agent_configs, os.path.join(scene_path, CONFIG_DIR, AGENT_CONFIG))


if __name__ == "__main__":
    args = parse_args()
    main(args)