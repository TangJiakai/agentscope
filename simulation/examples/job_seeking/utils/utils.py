import json
import importlib
import os
import glob
import re

from agentscope.file_manager import file_manager, _DEFAULT_CFG_NAME


def setup_agents(agent_configs, recent_n):
    """
    Load config and init agent by configs
    """
    with open(agent_configs, "r", encoding="utf-8") as file:
        configs = json.load(file)

    # get agent class
    def get_agent_cls(class_name):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        agent_files = glob.glob(os.path.join(parent_dir, 'agent', '*_agent.py'))

        for agent_file in agent_files:
            agent_module = os.path.basename(agent_file).replace('.py', '')
            module = importlib.import_module(f'agent.{agent_module}')
            if hasattr(module, class_name):
                class_ = getattr(module, class_name)
                return class_

    # setup agents
    agent_objs = []
    for config in configs:
        agent_cls = get_agent_cls(config["class"])
        agent_args = config["args"]
        agent = agent_cls(**agent_args)
        agent.recent_n = recent_n
        agent_objs.append(agent)
    return agent_objs


def extract_dict(text):
    pattern = re.compile(r'\{.*?\}', re.DOTALL)
    matches = pattern.findall(text)
    for match in matches:
        json_obj = json.loads(match)
        if json_obj:
            return json_obj
    raise text


def save_config(args):
    with open(
        os.path.join(file_manager.dir_root, _DEFAULT_CFG_NAME),
        "r+",
        encoding="utf-8",
    ) as file:
        cfg = json.load(file)
        cfg.update(vars(args))
        file.seek(0)
        json.dump(cfg, file, indent=4)
        file.truncate()


