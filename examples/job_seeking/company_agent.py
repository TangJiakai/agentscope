from typing import Optional
from jinja2 import Environment, FileSystemLoader
import json

from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.models.response import ModelResponse

file_loader = FileSystemLoader("prompts")
env = Environment(loader=file_loader)
Template = env.get_template('company_prompts.j2').module


class Company(object):
    """company object."""
    def __init__(self, id: int, name: str, cd: str, job_ids: list):
        self.id = id
        self.name = name
        self.cd = cd
        self.job_ids = job_ids

    def __str__(self):
        return (
            f"Company Name: {self.name}\n"
            f"Company Description: {self.cd}"
        )
    

class CompanyAgent(AgentBase):
    """company agent."""

    name: str   # Name of the company
    model_config_name: str  # Model config name
    company: Company  # Company object
    system_prompt: Msg  # System prompt

    def __init__(
        self,
        name: str,
        model_config_name: str,
        id: int,
        cd: str,
        job_ids: list
    ) -> None:
        super().__init__(
            name=name,
            model_config_name=model_config_name
        )
        self.company = Company(id, name, cd, job_ids)
        self.system_prompt = Msg("system", Template.system_prompt(self.company), role="system")

    def get_id(self):
        return self.company.id
    
    def update_fun(self):
        pass

    def reply(self, x: Optional[dict] = None) -> dict:
        return Msg(self.name, None, role="assistant")