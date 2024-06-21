import json
import importlib
import os
import glob
import re


def setup_agents(agent_configs):
    """
    Load config and init agent by configs
    """
    with open(agent_configs, "r", encoding="utf-8") as file:
        configs = json.load(file)

    # get agent class
    def get_agent_cls(class_name):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        agent_files = glob.glob(os.path.join(parent_dir, '*_agent.py'))

        for agent_file in agent_files:
            agent_module = os.path.basename(agent_file).replace('.py', '')
            module = importlib.import_module(agent_module)
            if hasattr(module, class_name):
                class_ = getattr(module, class_name)
                return class_

    # setup agents
    agent_objs = []
    for config in configs:
        # agent_cls = getattr(agent, config["class"])
        agent_cls = get_agent_cls(config["class"])
        agent_args = config["args"]
        agent = agent_cls(**agent_args)
        agent_objs.append(agent)
    return agent_objs


def extract_ids(input_str):
    """
    Extract ids from input string
    """
    pattern = r"\[(\d+)\]"
    ids = re.findall(pattern, input_str)
    return ids