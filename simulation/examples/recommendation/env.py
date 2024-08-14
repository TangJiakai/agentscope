from typing import List

import faiss
import numpy as np
from loguru import logger
from agentscope.agents import RpcAgent
from agentscope.environment import BasicEnv

from simulation.helpers.utils import *
from simulation.helpers.emb_service import *


class RecommendationEnv(BasicEnv):
    def __init__(
        self,
        name: str,
        embedding_api: str,
        item_infos: list,
        **kwargs,
    ) -> None:
        super().__init__(name=name)
        self.item_infos = item_infos
        self.embedding_api = embedding_api
        self.index = self._build_index(item_infos)
        self.all_agents: list[RpcAgent] = None

    def __getstate__(self) -> object:
        state = super().__getstate__()
        state["index"] = faiss.serialize_index(self.index)
        return state

    def __setstate__(self, state) -> None:
        if "index" in state:
            state["index"] = faiss.deserialize_index(state["index"])
        super().__setstate__(state)

    def _build_index(self, item_infos):
        logger.info("Building index......")
        item_embs = [
            get_embedding(
                item_info["title"] + item_info["genre"] + item_info["description"],
                self.embedding_api,
            )
            for item_info in item_infos
        ]
        index = faiss.IndexFlatL2(get_embedding_dimension(self.embedding_api))
        index.add(np.array(item_embs))
        logger.info("Index built!")
        return index

    def recommend4user(self, user_info, k=5):
        user_emb = get_embedding(user_info, self.embedding_api)
        _, indices = self.index.search(np.array([user_emb]), k)
        return get_assistant_msg([
            "\n".join([f"{k}: {v}" for k, v in self.item_infos[i].items()])
            for i in indices[0]
        ])

    def set_attr(self, attr: str, value, **kwargs) -> str:
        setattr(self, attr, value)
        return "success"

    def get_agents_by_ids(self, agent_ids: List[str]) -> List[RpcAgent]:
        agents = [agent for agent in self.all_agents if agent.agent_id in agent_ids]
        return agents
