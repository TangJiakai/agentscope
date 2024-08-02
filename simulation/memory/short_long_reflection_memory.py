# -*- coding: utf-8 -*-
import re
import os
from typing import Optional, List
from jinja2 import Environment, FileSystemLoader

from agentscope.models import ModelResponse
from agentscope.message import Msg
from agentscope.message import Msg

from simulation.memory.short_long_memory import ShortLongMemory

file_loader = FileSystemLoader(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
)
env = Environment(loader=file_loader)
Template = env.get_template("prompts.j2").module


class ShortLongReflectionMemory(ShortLongMemory):
    def __init__(
        self,
        *,
        embedding_size: int = 768,
        importance_weight: Optional[float] = 0.15,
        stm_K: int = 2,
        ltm_K: int = 2,
        reflection_threshold: Optional[float] = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            embedding_size=embedding_size,
            importance_weight=importance_weight,
            stm_K=stm_K,
            ltm_K=ltm_K,
            **kwargs,
        )
        self.reflection_threshold = reflection_threshold
        self.reflecting = False
        self.aggregate_importance = 0.0

    def _get_topics_of_reflection(self, last_k: int = 50):
        msg = Msg(
            "user",
            Template.get_topics_of_reflection_prompt(
                [_.content for _ in self.ltm_memory[-last_k:]]
            ),
            role="user",
        )
        prompt = self.model.format(msg)

        def parse_func(response: ModelResponse) -> ModelResponse:
            try:
                text = response.text
                lines = re.split(r"\n", text.strip())
                lines = [line for line in lines if line.strip()]  # remove empty lines
                res = [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]
                return ModelResponse(raw=res)
            except:
                raise ValueError(
                    f"Invalid response format in parse_func "
                    f"with response: {response.text}",
                )

        response = self.model(prompt, parse_func=parse_func).raw
        return response

    def _get_insights_on_topic(self, topic: Msg) -> List[str]:
        retrieved_memories = self.get_ltm_memory(topic)
        msg = Msg(
            "user",
            Template.get_insights_on_topic_prompt(retrieved_memories, topic, True),
            role="user",
        )
        prompt = self.model.format(msg)

        def parse_func(response: ModelResponse) -> ModelResponse:
            try:
                text = response.text
                lines = re.split(r"\n", text.strip())
                lines = [line for line in lines if line.strip()]  # remove empty lines
                res = [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]
                return ModelResponse(raw=res)
            except:
                raise ValueError(
                    f"Invalid response format in parse_func "
                    f"with response: {response.text}",
                )

        response = self.model(prompt, parse_func=parse_func).raw
        return response

    def pause_to_reflect(self):
        topics = self._get_topics_of_reflection()
        for topic in topics:
            insights = self._get_insights_on_topic(Msg("user", topic, role="user"))
            for insight in insights:
                self.add_ltm_memory(Msg("user", insight, role="user"))

    def add(self, memory: Msg = None):
        if memory is None: return

        super().add(memory)
        if len(self.ltm_memory) == 0:
            return

        self.aggregate_importance += self.ltm_memory[-1].importance_score
        if (
            self.aggregate_importance >= self.reflection_threshold
            and not self.reflecting
        ):
            self.reflecting = True
            self.pause_to_reflect()
            self.aggregate_importance = 0.0
            self.reflecting = False

    def get_memory(self, query: Msg = None):
        return super().get_memory(query)
