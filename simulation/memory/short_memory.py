# -*- coding: utf-8 -*-
from agentscope.message import Msg


class ShortMemory:
    def __init__(
        self,
        *,
        stm_K: int = 5,
        **kwargs,
    ) -> None:
        self.stm_K = stm_K
        self.stm_memory = []

    def add(self, memory: Msg = None):
        if memory is None: return None

        self.stm_memory.append(memory)
        if len(self.stm_memory) > self.stm_K:
            return self.stm_memory.pop(0)
        return None

    def get_memory(self, query: Msg = None):
        return self.stm_memory
