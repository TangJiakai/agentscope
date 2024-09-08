import os
import requests
from typing import List
from jinja2 import Environment, FileSystemLoader
import random
from loguru import logger

from agentscope.rpc import async_func

from simulation.helpers.utils import *
from simulation.helpers.constants import *
from simulation.helpers.base_agent import BaseAgent
from simulation.helpers.base_env import BaseEnv


scene_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_loader = FileSystemLoader(os.path.join(scene_path, "prompts"))
env = Environment(loader=file_loader)
Template = env.get_template("interviewer_prompts.j2").module


InterviewerAgentStates = [
    "idle",
    "screening cv",
    "making decision",
    "interviewing",
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
    def __init__(self, 
                name: str, 
                jd: str, 
                jr: List[str], 
                company: str,
                salary: str,
                benefits: List[str],
                location: str,):
        super().__init__(name=name, jd=jd, jr=jr, company=company, salary=salary, benefits=benefits, location=location)
        self.name = name
        self.jd = jd
        self.jr = jr
        self.company = company
        self.salary = salary
        self.benefits = benefits
        self.location = location

    def __str__(self):
        jr_string = "\n".join([f"- {r}" for r in self.jr])
        benefits_string = "\n".join([f"- {b}" for b in self.benefits])
        return (
            f"Position Name: {self.name}\n"
            f"Job Description: {self.jd}\n"
            f"Job Requirements:\n{jr_string}"
            f"Company: {self.company}\n"
            f"Salary: {self.salary}\n"
            f"Benefits:\n{benefits_string}"
            f"Location: {self.location}\n"
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
        company: str,
        salary: str,
        benefits: List[str],
        location: str,
        embedding: list,
        env: BaseEnv,
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

        self.memory.get_tokennum_func = self.get_tokennum_func

        self.job = Job(name=name, jd=jd, jr=jr, company=company, salary=salary, benefits=benefits, location=location)
        self.embedding = embedding
        self.env = env

        self.update_profile()
        self._state = "idle"

    def update_profile(self):
        self._profile = self.job.__str__()

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_value):
        pass
        # if hasattr(self, "backend_server_url"):
        #     if new_value not in InterviewerAgentStates:
        #         raise ValueError(f"Invalid state: {new_value}")
        #     self._state = new_value
        #     url = f"{self.backend_server_url}/api/state"
        #     resp = requests.post(
        #         url, json={"agent_id": self.agent_id, "state": new_value}
        #     )
        #     if resp.status_code != 200:
        #         logger.error(f"Failed to set state: {self.agent_id} -- {new_value}")

    def get_attr(self, attr):
        logger.info(f"Getting attribute: {attr}")
        if attr == "job":
            job = {
                "Position Name": self.job.name,
                "Job Description": self.job.jd,
                "Job Requirements": self.job.jr,
                "Company": self.job.company,
                "Salary": self.job.salary,
                "Benefits": self.job.benefits,
                "Location": self.job.location,
            }
            return job
        return super().get_attr(attr)

    @async_func
    @set_state("screening cv")
    def screening_cv(self, seeker_info: str):
        msg = get_assistant_msg()
        msg.instruction = Template.screening_cv_instruction()
        guided_choice = ["yes", "no"]
        msg.observation = Template.screening_cv_observation(seeker_info, guided_choice)
        content = self.reply(msg).content
        prompt = Template.parse_value_observation(content, guided_choice)
        reponse = self.model(self.model.format(get_assistant_msg(prompt))).text
        answer = random.choice(guided_choice)
        for c in guided_choice:
            if c in reponse:
                answer = c
                break
        return answer

    @async_func
    @set_state("interviewing")
    def interview(self, dialog: str):
        instruction = Template.interview_closing_instruction()
        guided_choice = ["yes", "no"]
        observation = Template.make_interview_decision_observation(dialog, guided_choice)
        msg = get_assistant_msg()
        msg.instruction = instruction
        msg.observation = observation
        content = self.reply(msg).content
        prompt = Template.parse_value_observation(content, guided_choice)
        reponse = self.model(self.model.format(get_assistant_msg(prompt))).text
        answer = random.choice(guided_choice)
        for c in guided_choice:
            if c in reponse:
                answer = c
                break
        return answer

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