import faiss
from loguru import logger
import numpy as np

from simulation.helpers.utils import *
from simulation.helpers.base_agent import BaseAgent
from simulation.helpers.emb_service import *

class EnvironmentAgent(BaseAgent):
    """environment agent."""

    def __init__(
        self,
        name: str,
        embedding_api: str,
        item_infos: list,
        **kwargs,
    ) -> None:
        super().__init__(name)
        self.item_infos = item_infos
        self.embedding_api = embedding_api
        self.index = self._build_index(item_infos)
        self.agent_distribution_infos = None

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
                self.embedding_api
            )
            for item_info in item_infos
        ]
        index = faiss.IndexFlatL2(get_embedding_dimension(self.embedding_api))
        index.add(np.array(item_embs))
        logger.info("Index built!")
        return index
    
    def recommend4user_fun(self, user_info, k=5):
        user_emb = get_embedding(user_info, self.embedding_api)
        _, indices = self.index.search(np.array([user_emb]), k)
        return get_assistant_msg([
            "\n".join([f"{k}: {v}" for k, v in self.item_infos[i].items()])
            for i in indices[0] 
        ])

    def get_agent_distribution_infos_fun(self, agent_ids: list):
        agent_infos = {
            agent_id: self.agent_distribution_infos[agent_id]
            for agent_id in agent_ids
        } 
        return get_assistant_msg(agent_infos)
        
    def run_fun(self, **kwargs):
        return get_assistant_msg("Done")
