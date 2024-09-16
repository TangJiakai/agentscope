import faiss
import numpy as np
from loguru import logger

from simulation.helpers.utils import *
from simulation.helpers.emb_service import *
from simulation.helpers.base_env import BaseEnv


class RecommendationEnv(BaseEnv):
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
        self.all_agents = None

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
        return [self.item_infos[i] for i in indices[0]]