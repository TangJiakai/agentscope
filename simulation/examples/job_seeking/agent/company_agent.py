import os
from typing import Optional
from jinja2 import Environment, FileSystemLoader
from typing import Union
from typing import Sequence

from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.models import load_model_by_config_name

scene_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_loader = FileSystemLoader(os.path.join(scene_path, "prompts"))
env = Environment(loader=file_loader)
Template = env.get_template('company_prompts.j2').module


class Company(object):
    """company object."""
    def __init__(self, name: str, cd: str):
        self.name = name
        self.cd = cd

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
        cd: str,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            model_config_name=model_config_name
        )
        self.model_config_name = model_config_name
        self.company = Company(name, cd)
        self.system_prompt = Msg("system", Template.system_prompt(self.company), role="system")

    def set_id(self, id: int):
        self.id = id

    def get_id(self):
        return self.id

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        state.pop("model")
        return state

    def __setstate__(self, state: object) -> None:
        self.__dict__.update(state)
        self.model = load_model_by_config_name(self.model_config_name)
    
    def reply(self, x: Optional[Union[Msg, Sequence[Msg]]] = None) -> Msg:
        return Msg(self.name, None, role="assistant")