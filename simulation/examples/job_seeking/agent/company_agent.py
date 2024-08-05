import os
import requests
from typing import Optional
from jinja2 import Environment, FileSystemLoader
from typing import Union
from typing import Sequence

from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.models import load_model_by_config_name
from loguru import logger

from simulation.helpers.message import MessageUnit
from simulation.helpers.utils import *

scene_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_loader = FileSystemLoader(os.path.join(scene_path, "prompts"))
env = Environment(loader=file_loader)
Template = env.get_template("company_prompts.j2").module


CompanyAgentStates = [
    "idle",
    "external interviewing",
]


backend_server_url = "http://localhost:39000"


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


class CompanyAgent(AgentBase):
    """company agent."""

    def __init__(
        self,
        name: str,
        model_config_name: str,
        memory_config: dict,
        embedding_api: str,
        cd: str,
        embedding: list,
        env_agent: AgentBase,
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

        self.sys_prompt = Msg("system", Template.sys_prompt(self.company), role="system")
        self._state = "idle"

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        state.pop("model")
        memory_state = self.memory.__dict__.copy()
        memory_state["model"] = None
        state["memory"] = memory_state
        return state

    def __setstate__(self, state: object) -> None:
        self.__dict__.update(state)
        self.memory = setup_memory(self.memory_config)
        self.memory.__dict__.update(state["memory"])
        self.model = load_model_by_config_name(self.model_config_name)
        self.memory.model = self.model

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_value):
        if new_value not in CompanyAgentStates:
            raise ValueError(f"Invalid state: {new_value}")
        self._state = new_value
        url = f"{backend_server_url}/api/state"
        resp = requests.post(url, json={"agent_id": self.agent_id, "state": new_value})
        if resp.status_code != 200:
            logger.error(f"Failed to set state: {self.agent_id} -- {new_value}")

    def _send_message(self, prompt, response):
        url = f"{backend_server_url}/api/message"
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

    def get_attr_fun(self, attr):
        if attr == "sys_prompt":
            return self.sys_prompt
        return get_assistant_msg(getattr(self, attr))

    @set_state("external interviewing")
    def external_interview_fun(self, query, **kwargs):
        query_msg = Msg("assistant", query, role="assistant")
        memory_msg = self.memory.get_memory(query_msg)
        msg = Msg(
            "assistant", "\n".join([p.content for p in memory_msg]) + query, "assistant"
        )
        prompt = self.model.format(self.sys_prompt, msg)
        response = self.model(prompt)
        return response.text

    def run_fun(self, **kwargs):
        return Msg("assistant", "Done", role="assistant")

    def reply(self, x: Optional[Union[Msg, Sequence[Msg]]] = None) -> Msg:
        if x and x.get("fun", None):
            return getattr(self, f"{x.fun}_fun")(**getattr(x, "params", {}))
        else:
            memory = self.memory.get_memory(x)
            if x:
                self.memory.add(x)
                msg = Msg(
                    "user",
                    "\n".join([p["content"] for p in memory]) + x["content"],
                    "user",
                )
            else:
                msg = Msg("user", "\n".join([p["content"] for p in memory]), "user")
            prompt = self.model.format(self.sys_prompt, msg)
            response = self.model(prompt)
            self._send_message(prompt, response)
            msg = Msg(self.name, response.text, role="user")
            self.memory.add(msg)
            return msg
