from typing import List, Optional
from pydantic import BaseModel
import queue


class MessageUnit(BaseModel):
    msg_id: Optional[int] = None
    agent_id: str
    name: str
    agent_type: str
    prompt: str
    completion: str
    rewritten_response: Optional[str] = None
    rating: Optional[int] = None


class StateUnit(BaseModel):
    agent_id: str
    state: str


class MessageManager:
    def __init__(self):
        self.messages: List[MessageUnit] = []
        self.message_queue = queue.Queue()
        self.state_queue = queue.Queue()

    def add_message(self, message: MessageUnit):
        from backend.app import lock

        with lock:
            self.messages.append(message)
        self.message_queue.put(message)

    def add_state(self, state: StateUnit):
        self.state_queue.put(state)

    def clear(self):
        from backend.app import lock

        with lock:
            self.messages.clear()
        self.message_queue = queue.Queue()
        self.state_queue = queue.Queue()


message_manager = MessageManager()
