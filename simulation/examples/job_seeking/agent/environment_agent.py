import os
from typing import Optional
from jinja2 import Environment, FileSystemLoader
from typing import Union
from typing import Sequence

from agentscope.agents import AgentBase
from agentscope.message import Msg

from simulation.helpers.utils import *

scene_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_loader = FileSystemLoader(os.path.join(scene_path, "prompts"))
env = Environment(loader=file_loader)
Template = env.get_template('environment_prompts.j2').module


class EnvironmentAgent(AgentBase):
    """environment agent."""

    def __init__(
        self,
        name: str,
        **kwargs,
    ) -> None:
        super().__init__(name)
        self.agent_distribution_infos = None

    def set_agent_distribution_infos_fun(self, agent_distribution_infos: dict):
        self.agent_distribution_infos = agent_distribution_infos
        return get_assistant_msg("success")

    def get_agent_distribution_infos_fun(self, agent_ids: list):
        agent_infos = {
            agent_id: self.agent_distribution_infos[agent_id]
            for agent_id in agent_ids
        } 
        return get_assistant_msg(agent_infos)
        
    def run_fun(self, **kwargs):
        return get_assistant_msg("Done")

    def reply(self, x: Optional[Union[Msg, Sequence[Msg]]] = None) -> Msg:
        return getattr(self, f"{x.fun}_fun")(**getattr(x, "params", {}))
        