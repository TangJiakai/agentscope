from typing import List, Any, Union, Mapping
from copy import deepcopy
import re
import random
import threading
import time
from loguru import logger

from simulation.helpers.utils import *
from simulation.helpers.emb_service import *
from simulation.helpers.base_env import BaseEnv

from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.exception import (
    EnvListenerError,
)
from agentscope.environment import (
    Env,
    BasicEnv,
    EventListener,
    Event,
    event_func,
)
from agentscope.models import ModelResponse


class ChatRoomMember(BasicEnv):
    """A member of chatroom."""

    def __init__(
        self,
        name: str,
        agent: AgentBase,
        history_idx: int = 0,
    ) -> None:
        super().__init__(name)
        self._agent = agent
        self._history_idx = history_idx

    @property
    def agent_name(self) -> str:
        """Get the name of the agent."""
        return self._agent.name

    @property
    def history_idx(self) -> int:
        """Get the history index of the agent."""
        return self._history_idx

    @property
    def agent(self) -> AgentBase:
        """Get the agent of the member."""
        return self._agent

    def chatting(self, delay: int = 1) -> None:
        """Make the agent chatting in the chatroom."""
        time.sleep(delay)
        while True:
            msg = self._agent()
            if "goodbye" in msg.content.lower():
                break
            sleep_time = random.randint(1, 5)
            time.sleep(sleep_time)


class ChatRoom(BaseEnv):
    """A chatroom env."""

    def __init__(
        self,
        name: str = None,
        announcement: Msg = None,
        participants: List[AgentBase] = None,
        all_history: bool = False,
        **kwargs: Any,
    ) -> None:
        """Init a ChatRoom instance.

        Args:
            name (`str`): The name of the chatroom.
            announcement (`Msg`): The announcement message.
            participants (`List[AgentBase]`): A list of agents
            all_history (`bool`): If `True`, new participant can see all
            history messages, else only messages generated after joining
            can be seen. Default to `False`.
        """
        super().__init__(
            name=name,
            **kwargs,
        )
        self.children = {}
        for p in participants if participants else []:
            self.join(p)
        self.event_listeners = {}
        self.all_history = all_history
        self.history = []
        self.announcement = announcement

    @event_func
    def join(self, agent: AgentBase) -> bool:
        """Add a participant to the chatroom."""
        if agent.agent_id in self.children:
            return False
        self.children[agent.agent_id] = ChatRoomMember(
            name=agent.agent_id,
            agent=agent,
            history_idx=len(self.history),
        )
        self.add_listener("speak", Mentioned(agent))
        return True

    @event_func
    def leave(self, agent: AgentBase) -> bool:
        """Remove the participant agent from the chatroom."""
        if agent.agent_id not in self.children:
            return False
        del self.children[agent.agent_id]
        return True

    @event_func
    def speak(self, message: Msg) -> None:
        """Speak a message in the chatroom."""
        self.history.append(message)

    @event_func
    def get_history(self, agent_id: str) -> List[Msg]:
        """Get all history messages, since the participant join in the
        chatroom"""
        if agent_id not in self.children:
            # only participants can get history message
            return []
        if self.all_history:
            history_idx = 0
        else:
            history_idx = self.children[agent_id].history_idx
        return deepcopy(self.history[history_idx:])

    @event_func
    def set_announcement(self, announcement: Msg) -> None:
        """Set the announcement of the chatroom."""
        self.announcement = announcement

    @event_func
    def get_announcement(self) -> Msg:
        """Get the announcement of the chatroom."""
        return deepcopy(self.announcement)

    # Syntaic sugar, not an event function
    def listen_to(
        self,
        target_names: List[str],
        listener: EventListener,
    ) -> None:
        """The listener will be called when a message whose name is in
        `target_names` is send to the chatroom."""
        if target_names is None or len(target_names) == 0:
            return

        class ListenTo(EventListener):
            """A middleware that activates `target_listener`"""

            def __init__(
                self,
                name: str,
                target_names: List[str],
                target_listener: EventListener,
            ) -> None:
                super().__init__(name=name)
                self.target_names = target_names
                self.target_listener = target_listener

            def __call__(self, env: Env, event: Event) -> None:
                if event.args["message"].name in self.target_names:
                    self.target_listener(env, event)

        if not self.add_listener(
            "speak",
            listener=ListenTo(
                name=f"listen_to_{listener.name}",
                target_names=target_names,
                target_listener=listener,
            ),
        ):
            raise EnvListenerError("Fail to add listener.")

    def chatting_parse_func(self, response: ModelResponse) -> ModelResponse:
        """Parse the response of the chatting agent."""
        pattern_str = ""
        for child in self.children.values():
            if pattern_str:
                pattern_str += "|"
            pattern_str += rf"""\s?{child.agent_name}: """
        pattern = re.compile(pattern_str, re.DOTALL)
        logger.debug(repr(pattern_str))
        logger.debug(response.text)
        texts = [s.strip() for s in pattern.split(response.text)]
        logger.debug(texts)
        return ModelResponse(text=texts[0])

    def chatting(self, delay: Union[int, Mapping[str, int]] = 1) -> None:
        """Make all agents chatting in the chatroom."""
        tasks = []
        for agent_id, child in self.children.items():
            if isinstance(delay, int):
                tasks.append(
                    threading.Thread(target=child.chatting, args=(delay,)),
                )
            else:
                if agent_id not in delay:
                    continue
                tasks.append(
                    threading.Thread(
                        target=child.chatting,
                        args=(delay[agent_id],),
                    ),
                )
        for task in tasks:
            task.start()
        for task in tasks:
            task.join()


class Mentioned(EventListener):
    """A listener that will be called when a message is mentioned the agent"""

    def __init__(
        self,
        agent: AgentBase,
    ) -> None:
        super().__init__(name=f"mentioned_agent_{agent.name}")
        self.agent = agent
        self.pattern = re.compile(r"""(?<=@)\w*""", re.DOTALL)

    def __call__(self, env: Env, event: Event) -> None:
        find_result = self.pattern.findall(str(event.args["message"].content))
        if self.agent.name in find_result:
            logger.info(
                f"{event.args['message'].name} mentioned {self.agent.name}.",
            )
            self.agent.add_mentioned_message(event.args["message"])
