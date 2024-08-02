# -*- coding: utf-8 -*-
from agentscope.message import (Msg)


class NoneMemory:
    def __init__(
        self,
        **kwargs,
    ) -> None:
        pass

    def add(self, memory: Msg = None):
        pass

    def get_memory(self, query: Msg = None):
        return []



        