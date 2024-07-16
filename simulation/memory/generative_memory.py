import os
import re
from typing import Callable
from typing import Union
from typing import Sequence
from jinja2 import Environment, FileSystemLoader
from typing import Any, List, Optional
from datetime import datetime
from loguru import logger

from agentscope.memory import TemporaryMemory
from agentscope.models import load_model_by_config_name, ModelWrapperBase
from agentscope.message import (
    MessageBase,
    Msg,
    PlaceholderMessage,
)
from agentscope.models import ModelResponse
from agentscope.service.retrieval.similarity import Embedding
from agentscope.service import cos_sim
from agentscope.service.service_response import ServiceResponse
from agentscope.service.service_status import ServiceExecStatus

file_loader = FileSystemLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts"))
env = Environment(loader=file_loader)
Template = env.get_template('generative_memory_prompts.j2').module


def retrieve_from_list(
    query: Any,
    knowledge: Sequence,  # TODO: rename
    score_func: Callable[[Any, Any], float],
    top_k: int = None,
    embedding_model: Optional[ModelWrapperBase] = None,
    preserve_order: bool = True,
) -> ServiceResponse:
    if isinstance(query, dict):
        if embedding_model is not None and "embedding" not in query:
            query["embedding"] = embedding_model(query.content).embedding[0]
        elif embedding_model is None and "embedding" not in query:
            logger.warning(
                "Since the input query has no embedding, embedding model is "
                "is not provided either.",
            )

    # (score, index, object)
    scores = [
        (score_func(query, msg), i, msg) for i, msg in enumerate(knowledge)
    ]

    # ordered by score, and extract the top-k items with highest scores
    top_k = len(scores) if top_k is None else top_k
    ordered_top_k_scores = sorted(scores, key=lambda x: x[0], reverse=True)[
        :top_k
    ]

    # if keep the original order
    if preserve_order:
        # ordered by index
        content = sorted(ordered_top_k_scores, key=lambda x: x[1])
    else:
        content = ordered_top_k_scores

    # The returned content includes a list of triples of (score, index, object)
    return ServiceResponse(
        status=ServiceExecStatus.SUCCESS,
        content=content,
    )


class GenerativeMemory(TemporaryMemory):
    def __init__(
        self,
        model: Union[str, Callable] = None,
        config: Optional[dict] = None,
        embedding_model: Union[str, Callable] = None,
        recent_n: int = 5,
        reflection_threshold: Optional[float] = 5.0,
        importance_weight: float = 0.15,
        aggregate_importance: float = 0.0,  # : :meta private:
        reflecting: bool = False,
    ) -> None:
        """
        Temporary memory module for conversation.
        Args:
            config (dict):
                configuration of the memory
            embedding_model (Union[str, Callable])
                if the temporary memory needs to be embedded,
                then either pass the name of embedding model or
                the embedding model itself.
        """
        super().__init__(config)

        self._content = []

        if isinstance(model, str):
            self.model = load_model_by_config_name(model)
        else:
            self.model = model

        # prepare embedding model if needed
        if isinstance(embedding_model, str):
            self.embedding_model = load_model_by_config_name(embedding_model)
        else:
            self.embedding_model = embedding_model

        self.recent_n = recent_n
        self.reflection_threshold = reflection_threshold
        self.importance_weight = importance_weight
        self.aggregate_importance = aggregate_importance
        self.reflecting = reflecting

    def _score_memory_importance(self, memory_content: str) -> List[float]:
        msg = Msg("user", Template.score_importance_prompt(memory_content), role="user")
        prompt = self.model.format(msg)

        def parse_func(response: ModelResponse) -> ModelResponse:
            try:
                match = re.search(r"^\D*(\d+)", response.text.strip())
                if match:
                    res = (float(match.group(1)) / 10) * self.importance_weight
                else:
                    res = 0.0
                return ModelResponse(raw=res)
            except:
                raise ValueError(
                    f"Invalid response format in parse_func "
                    f"with response: {response.text}",
                )

        response = self.model(prompt, parse_func=parse_func).raw
        return response

    def _get_topics_of_reflection(self, last_k: int = 50):
        """Return the 3 most salient high-level questions about recent observations."""
        memory_detail = Template.format_memories_detail_prompt(self._content[-last_k:])
        msg = Msg("user", Template.get_topics_of_reflection_prompt(memory_detail), role="user")
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

    def retrieve_by_embedding(
        self,
        query: Union[str, Embedding],
        preserve_order: bool = True,
    ) -> list[dict]:
        def score_func(m1: MessageBase, m2: MessageBase) -> float:
            relevance = cos_sim(m1.embedding, m2.embedding).content
            time_gap = (
                datetime.strptime(m1.timestamp, "%Y-%m-%d %H:%M:%S")
                - datetime.strptime(m2.timestamp, "%Y-%m-%d %H:%M:%S")
            ).total_seconds() / 60
            recency = 0.99**time_gap
            return recency + relevance
        
        query = Msg(name="assistant", content=query, role="assistant")
        
        retrieved_items = retrieve_from_list(
            query,
            list(self.get_memory(self.recent_n)),
            score_func,
            None,
            self.embedding_model,
            preserve_order,
        ).content

        # obtain the corresponding memory item
        response = []
        for score, index, _ in retrieved_items:
            response.append(
                {
                    "score": score + self._content[index].importance,
                    "index": index,
                    "memory": self._content[index],
                },
            )

        response = sorted(response, key=lambda x: x["score"])[-self.recent_n:]
        return [item["memory"] for item in response]
    
    def _get_insights_on_topic(self, topic: str) -> List[str]:
        retrieved_memories = self.retrieve_by_embedding(topic)
        msg = Msg("user", Template.get_insights_on_topic_prompt(retrieved_memories, topic), role="user")
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

    def pause_to_reflect(self) -> List[str]:
        new_insights = []
        topics = self._get_topics_of_reflection()
        for topic in topics:
            insights = self._get_insights_on_topic(topic)
            for insight in insights:
                self.add(Msg("assistant", insight, role="assistant"))
            new_insights.extend(insights)
        return new_insights
    
    def get_embeddings(self) -> list:
        embeddings = []
        for memory_unit in self._content:
            if memory_unit.embedding is None:
                memory_unit.embedding = self.embedding_model(memory_unit.content).embedding
            embeddings.append(memory_unit.embedding)
        return embeddings
    
    def add(
        self,
        memories: Union[Sequence[Msg], Msg, None],
        embed: bool = True,
    ) -> None:

        if memories is None:
            return

        if not isinstance(memories, Sequence):
            record_memories = [memories]
        else:
            record_memories = memories

        memories_idx = set(_.id for _ in self._content if hasattr(_, "id"))
        max_importance_score = 0.0
        for memory_unit in record_memories:
            if not issubclass(type(memory_unit), MessageBase):
                try:
                    memory_unit = Msg(**memory_unit)
                except Exception as exc:
                    raise ValueError(
                        f"Cannot add {memory_unit} to memory, "
                        f"must be with subclass of MessageBase",
                    ) from exc

            # in case this is a PlaceholderMessage, try to update
            # the values first
            if isinstance(memory_unit, PlaceholderMessage):
                memory_unit.update_value()
                memory_unit = Msg(**memory_unit)

            # add to memory if it's new
            if (
                not hasattr(memory_unit, "id")
                or memory_unit.id not in memories_idx
            ):
                if embed:
                    if self.embedding_model:
                        # TODO: embed only content or its string representation
                        memory_unit.embedding = self.embedding_model(memory_unit.content).embedding[0]
                    else:
                        raise RuntimeError("Embedding model is not provided.")

                importance_score = self._score_memory_importance(memory_unit.content)
                memory_unit.importance = importance_score
                self._content.append(memory_unit)

                max_importance_score = max(max_importance_score, importance_score)
            
        self.aggregate_importance += max_importance_score

        if (
            self.reflection_threshold is not None
            and self.aggregate_importance > self.reflection_threshold
            and not self.reflecting
        ):
            self.reflecting = True
            self.pause_to_reflect()
            self.aggregate_importance = 0.0
            self.reflecting = False