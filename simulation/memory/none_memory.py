# -*- coding: utf-8 -*-
from typing import Sequence, Union
from agentscope.message import Msg


class NoneMemory:
    def __init__(
        self,
        **kwargs,
    ) -> None:
        pass

    def __getstate__(self):
        return dict()

    def __setstate__(self, state):
        pass

    def add(self, memory: Union[Sequence[Msg], Msg, None] = None):
        pass

    def get_memory(self, query: Msg = None):
        return []
