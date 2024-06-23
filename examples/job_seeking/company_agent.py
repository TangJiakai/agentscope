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

    def reply(self, x: Optional[dict] = None) -> dict:
        return Msg(self.name, None, role="assistant")
        # if self.memory:
        #     self.memory.add(x)

        # msg_hint = Msg("system", HINT_PROMPT, role="system")

        # prompt = self.model.format(
        #     self.memory.get_memory(),
        #     msg_hint,
        # )

        # response = self.model(
        #     prompt,
        #     parse_func=parse_func,
        #     max_retries=3,
        # ).raw

        # # For better presentation, we print the response proceeded by
        # # json.dumps, this msg won't be recorded in memory
        # self.speak(
        #     Msg(
        #         self.name,
        #         json.dumps(response, indent=4, ensure_ascii=False),
        #         role="assistant",
        #     ),
        # )

        # if self.memory:
        #     self.memory.add(Msg(self.name, response, role="assistant"))

        # # Hide thought from the response
        # return Msg(self.name, response["move"], role="assistant")