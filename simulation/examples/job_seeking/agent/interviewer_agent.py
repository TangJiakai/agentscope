import os
import threading
import requests
from typing import List
from jinja2 import Environment, FileSystemLoader

from agentscope.message import Msg
from agentscope.rpc import async_func
from loguru import logger

from simulation.helpers.utils import *
from simulation.helpers.constants import *
from simulation.helpers.base_agent import BaseAgent

scene_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_loader = FileSystemLoader(os.path.join(scene_path, "prompts"))
env = Environment(loader=file_loader)
Template = env.get_template("interviewer_prompts.j2").module


InterviewerAgentStates = [
    "idle",
    "screening cv",
    "making decision",
    "receiving notification",
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
    def __init__(self, name: str, jd: str, jr: List[str]):
        super().__init__(name=name, jd=jd, jr=jr)
        self.name = name
        self.jd = jd
        self.jr = jr

    def __str__(self):
        jr_string = "\n".join([f"- {r}" for r in self.jr])
        return (
            f"Position Name: {self.name}\n"
            f"Job Description: {self.jd}\n"
            f"Job Requirements:\n{jr_string}"
        )


class InterviewerAgent(BaseAgent):
    """Interviewer agent."""

    def __init__(
        self,
        name: str,
        model_config_name: str,
        memory_config: dict,
        embedding_api: str,
        jd: str,
        jr: list,
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
        self.job = Job(name=name, jd=jd, jr=jr)
        self.embedding = embedding
        self.env_agent = env_agent

        self.update_profile()
        self._state = "idle"

    def update_profile(self):
        self._profile = self.job.__str__()

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

    def get_attr(self, attr):
        if attr == "job":
            job = {
                "Position Name": self.job.name,
                "Job Description": self.job.jd,
                "Job Requirements": self.job.jr,
            }
            return job
            # return get_assistant_msg(job)
        elif attr == "sys_prompt":
            return self.sys_prompt
        return super().get_attr(attr)

    @async_func
    @set_state("screening cv")
    def screening_cv(self, seeker_info: str):
        msg = get_assistant_msg()
        msg.instruction = Template.screening_cv_instruction()
        selection = ["yes", "no"]
        msg.observation = Template.make_choice_observation(selection)
        msg.selection_num = len(selection)
        response = selection[int(self.reply(msg)["content"])]
        return response
        # return get_assistant_msg(response)

    @set_state("interviewing")
    def interview(self, msg: Msg):
        observation = msg["content"]
        if hasattr(msg, "end") and msg.end:
            instruction = Template.interview_closing_instruction()
            selection = ["yes", "no"]
            observation = observation + Template.make_choice_observation(selection)
            msg = get_assistant_msg()
            msg.instruction = instruction
            msg.observation = observation
            msg.selection = selection
            response = selection[int(self.reply(msg)["content"])]
            return response
            # return get_assistant_msg(response)
        else:
            instruction = Template.interview_opening_instruction()
            format_instruction = PROFILE_BEGIN + instruction + PROFILE_END
            format_profile = PROFILE_BEGIN + self._profile + PROFILE_END
            memory = self.memory.get_memory(get_assistant_msg(instruction + observation))
            format_memory = MEMORY_BEGIN + "\n- ".join([m["content"] for m in memory]) + MEMORY_END
            format_observation = OBSERVATION_BEGIN + observation + OBSERVATION_END + "Interviewer:"
            response = self.model(self.model.format(Msg(
                "user",
                format_instruction + format_profile + format_memory + format_observation,
                role="user",
            )))
            return response.text
            # return get_assistant_msg(response.text)

    @async_func
    @set_state("receiving notification")
    def receive_notification(self, seeker_name: str, is_accept: bool, **kwargs):
        self.observe(
            get_assistant_msg(
                Template.receive_notification_observation(seeker_name, is_accept)
            )
        )
        return get_assistant_msg("sucesss")

    @async_func
    def run(self, **kwargs):
        return "Done"