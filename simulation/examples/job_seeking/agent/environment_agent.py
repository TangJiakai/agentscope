import os
from jinja2 import Environment, FileSystemLoader

from simulation.helpers.utils import *
from simulation.helpers.base_agent import BaseAgent


class EnvironmentAgent(BaseAgent):
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
