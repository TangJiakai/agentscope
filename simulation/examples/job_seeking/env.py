from typing import List

from agentscope.agents import RpcAgent
from agentscope.environment import BasicEnv


class JobSeekingEnv(BasicEnv):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name=name)
        self.all_agents: List[RpcAgent] = None

    def set_attr(self, attr: str, value, **kwargs) -> str:
        setattr(self, attr, value)
        return "success"

    def get_agents_by_ids(self, agent_ids: List[str]) -> List[RpcAgent]:
        agents = [agent for agent in self.all_agents if agent.agent_id in agent_ids]
        return agents
