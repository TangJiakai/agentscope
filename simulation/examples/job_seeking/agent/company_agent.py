import os
import requests
from jinja2 import Environment, FileSystemLoader

from agentscope.message import Msg
from loguru import logger

from simulation.helpers.utils import *
from simulation.helpers.constants import * 
from simulation.helpers.base_agent import BaseAgent

scene_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_loader = FileSystemLoader(os.path.join(scene_path, "prompts"))
env = Environment(loader=file_loader)
Template = env.get_template("company_prompts.j2").module


CompanyAgentStates = [
    "idle",
]


class Company(object):
    """company object."""

    def __init__(self, name: str, cd: str):
        self.name = name
        self.cd = cd

    def __str__(self):
        return f"Company Name: {self.name}\n" f"Company Description: {self.cd}"


def set_state(flag: str):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            init_state = self.state
            self.state = flag
            try:
                return func(self, *args, **kwargs)
            finally:
                self.state = init_state

        return wrapper

    return decorator


class CompanyAgent(BaseAgent):
    """company agent."""

    def __init__(
        self,
        name: str,
        model_config_name: str,
        memory_config: dict,
        embedding_api: str,
        cd: str,
        embedding: list,
        env_agent: BaseAgent,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            model_config_name=model_config_name,
        )
        self.model_config_name = model_config_name
        self.memory_config = memory_config
        self.embedding_api = embedding_api
        self.memory = setup_memory(memory_config)
        self.memory.model = self.model
        self.memory.embedding_api = embedding_api
        self.company = Company(name, cd)
        self.embedding = embedding
        self.env_agent = env_agent

        self.sys_prompt = Msg(
            "system", Template.sys_prompt(self.company), role="system"
        )
        self._update_profile()
        self._state = "idle"
    
    def _update_profile(self):
        self._profile = self.company.__str__()

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_value):
        if hasattr(self, "backend_server_url"):
            if new_value not in CompanyAgentStates:
                raise ValueError(f"Invalid state: {new_value}")
            self._state = new_value
            url = f"{self.backend_server_url}/api/state"
            resp = requests.post(
                url, json={"agent_id": self.agent_id, "state": new_value}
            )
            if resp.status_code != 200:
                logger.error(f"Failed to set state: {self.agent_id} -- {new_value}")

    def _send_message(self, prompt, response):
        if hasattr(self, "backend_server_url"):
            url = f"{self.backend_server_url}/api/message"
            resp = requests.post(
                url,
                json={
                    "name": self.name,
                    "prompt": "\n".join([p["content"] for p in prompt]),
                    "completion": response.text,
                    "agent_type": type(self).__name__,
                    "agent_id": self.agent_id,
                },
            )
            if resp.status_code != 200:
                logger.error(f"Failed to send message: {self.agent_id}")

    def run_fun(self, **kwargs):
        return Msg("assistant", "Done", role="assistant")

