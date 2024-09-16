# -*- coding: utf-8 -*-
from typing import Sequence, Union

from agentscope.message import Msg


class ShortMemory:
    def __init__(
        self,
        *,
        stm_K: int = 2,
        **kwargs,
    ) -> None:
        self.stm_K = stm_K
        self.stm_memory = []
        self.get_tokennum_func = None

    def add(self, memory: Union[Sequence[Msg], Msg, None] = None):
        if memory is None: return None

        if isinstance(memory, Msg):
            self.stm_memory.append(memory)
        elif isinstance(memory, Sequence):
            self.stm_memory.extend(memory)
        if len(self.stm_memory) > self.stm_K:
            return self.stm_memory.pop(0)
        return None

    def get_memory(self, query: Msg = None):
        return self.stm_memory
