import os
import threading
import requests
from typing import List, Optional
from jinja2 import Environment, FileSystemLoader
from typing import Union
from typing import Sequence

from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.manager import ModelManager
from loguru import logger

from simulation.helpers.utils import *

scene_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_loader = FileSystemLoader(os.path.join(scene_path, "prompts"))
env = Environment(loader=file_loader)
Template = env.get_template("interviewer_prompts.j2").module


InterviewerAgentStates = [
    "idle",
    "screening cv",
    "making decision",
    "receiving notification",
    "external interviewing",
]


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


class Job(dict):
    def __init__(self, name: str, jd: str, jr: List[str], hc: int):
        super().__init__(name=name, jd=jd, jr=jr, hc=hc)
        self.name = name
        self.jd = jd
        self.jr = jr
        self.hc = hc

    def __str__(self):
        jr_string = "\n".join([f"- {r}" for r in self.jr])
        return (
            f"Position Name: {self.name}\n"
            f"Job Description: {self.jd}\n"
            f"Job Requirements:\n{jr_string}\n"
            f"Headcount: {self.hc}"
        )


class InterviewerAgent(AgentBase):
    """Interviewer agent."""

    def __init__(
        self,
        name: str,
        model_config_name: str,
        memory_config: dict,
        embedding_api: str,
        jd: str,
        jr: list,
        hc: int,
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
        self.job = Job(name=name, jd=jd, jr=jr, hc=hc)
        self.embedding = embedding
        self.env_agent = env_agent

        self.update_sys_prompt()
        self._lock = threading.Lock()
        self._state = "idle"

    def update_sys_prompt(self):
        self.sys_prompt = Msg("system", Template.sys_prompt(self.job), role="system")

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
        self.model = ModelManager.get_instance().get_model_by_config_name(
            self.model_config_name
        )
        self.memory.model = self.model

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_value):
        if hasattr(self, "backend_server_url"):
            if new_value not in InterviewerAgentStates:
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

    def _acquire_lock(self):
        self._lock.acquire()

    def _release_lock(self):
        self._lock.release()

    def set_attr_fun(self, attr: str, value, **kwargs):
        setattr(self, attr, value)
        return get_assistant_msg("success")

    def get_attr_fun(self, attr):
        if attr == "job":
            job = {
                "Position Name": self.job.name,
                "Job Description": self.job.jd,
                "Job Requirements": self.job.jr,
                "Headcount": self.job.hc,
            }
            return get_assistant_msg(job)
        elif attr == "sys_prompt":
            return self.sys_prompt
        return get_assistant_msg(getattr(self, attr))

    @set_state("screening cv")
    def screening_cv_fun(self, seeker_info: str):
        if self.job.hc <= 0:
            return Msg(self.name, "No headcount available", role="assistant")

        msg = Msg(
            "user", Template.screening_cv_prompt(seeker_info, self.job.hc), role="user"
        )
        return self.reply(msg)

    def start_interview_fun(self, msg: Msg):
        self._acquire_lock()
        return self.reply(msg)

    def end_interview_fun(self, msg: Msg):
        response = self.reply(
            Msg(
                "assistant",
                msg["content"] + Template.interview_closing_statement(),
                role="assistant",
            )
        )
        self._release_lock()
        return response

    @set_state("receiving notification")
    def receive_notification_fun(self, seeker_name: str, is_accept: bool, **kwargs):
        self.observe(
            get_assistant_msg(
                Template.receive_notification_observation(seeker_name, is_accept)
            )
        )
        if is_accept:
            self.job.hc -= 1
            self.update_sys_prompt()
        return get_assistant_msg("sucesss")

    @set_state("external interviewing")
    def external_interview_fun(self, query, **kwargs):
        query_msg = get_assistant_msg(query)
        memory_msg = self.memory.get_memory(query_msg)
        msg = get_assistant_msg("\n".join([p["content"] for p in memory_msg]))
        prompt = self.model.format(self.sys_prompt, msg)
        response = self.model(prompt)
        return response.text

    def run_fun(self, **kwargs):
        return get_assistant_msg("Done")

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
