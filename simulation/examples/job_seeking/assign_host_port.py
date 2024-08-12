import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import argparse
from itertools import zip_longest
import math
import json

from simulation.helpers.utils import load_json
from simulation.helpers.constants import *

scene_path = os.path.dirname(os.path.abspath(__file__))
SEEKER_AGENT_CONFIG = "seeker_agent_configs.json"
INTERVIEWER_AGENT_CONFIG = "interviewer_agent_configs.json"

def parse_args() -> argparse.Namespace:
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--base_port", type=int, default=12010)
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

    seeker_configs = load_json(os.path.join(scene_path, CONFIG_DIR, SEEKER_AGENT_CONFIG))
    interview_configs = load_json(os.path.join(scene_path, CONFIG_DIR, INTERVIEWER_AGENT_CONFIG))

    print("len(seeker_configs):", len(seeker_configs))
    print("len(interview_configs):", len(interview_configs))

    total_agent_num = len(seeker_configs) + len(interview_configs)
    agent_num_per_server = math.ceil(total_agent_num / available_port_num)
    print("agent_num_per_server:", agent_num_per_server)

    interleaved_configs = [config for sublist in zip_longest(
        seeker_configs, interview_configs
        ) for config in sublist if config is not None]

    for i, agent_config in enumerate(interleaved_configs):
        agent_config["args"]["host"] = host
        agent_config["args"]["port"] = base_port + i // agent_num_per_server

    save_agent_configs(seeker_configs, os.path.join(scene_path, CONFIG_DIR, SEEKER_AGENT_CONFIG))
    save_agent_configs(interview_configs, os.path.join(scene_path, CONFIG_DIR, INTERVIEWER_AGENT_CONFIG))


if __name__ == "__main__":
    args = parse_args()
    main(args)