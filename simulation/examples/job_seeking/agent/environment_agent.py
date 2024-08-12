import os
from jinja2 import Environment, FileSystemLoader

from agentscope.agents import RpcAgent
from agentscope.rpc import async_func

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
        self.all_agents: list[RpcAgent] = None

    def get_agents_by_ids(self, agent_ids: list[str]):
        agents = [agent for agent in self.all_agents if agent.agent_id in agent_ids]
        return agents

    def get_agent_distribution_infos(self, agent_ids: list):
        agent_infos = {
            agent_id: self.agent_distribution_infos[agent_id]
            for agent_id in agent_ids
        } 
        # return get_assistant_msg(agent_infos)
        return agent_infos
    
    @async_func
    def run(self, **kwargs):
        return "Done"
