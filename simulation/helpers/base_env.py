from typing import Dict, List

from agentscope.environment import BasicEnv
from agentscope.rpc import async_func

from simulation.helpers.utils import *


class BaseEnv(BasicEnv):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name=name)
        self.all_agents: Dict = dict()

    @async_func
    def set_attr(self, attr: str, value, **kwargs) -> str:
        attrs = attr.split(".")
        obj = self
        for attr in attrs[:-1]:
            obj = getattr(obj, attr)
        setattr(obj, attrs[-1], value)
        return "success"
    
    def get_agents_by_ids(self, agent_ids: List[str]):
        agents = {agent_id: self.all_agents[agent_id] for agent_id in agent_ids}
        return agents

    def broadcast(self, content: str) -> None:
        # TODO: currently wait for each agent to finish processing the message (slow & unreasonable)
        for agent in self.all_agents.values():
            agent.observe(get_assistant_msg(content))

    def intervention(self, agent_id: str, key, value) -> None:
        if agent_id in self.all_agents:
            agent = self.all_agents[agent_id]
            agent.set_attr(key, value).get()

    def interview(self, agent_id: str, query: str) -> str:
        if agent_id in self.all_agents:
            agent = self.all_agents[agent_id]
            return agent.external_interview(query)