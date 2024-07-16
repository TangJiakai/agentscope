from pydantic import BaseModel
import queue 


class MessageUnit(BaseModel):
    round: int
    name: str
    prompt: str
    completion: str
    agent_type: str
    agent_id: int


class MessageManager:
    def __init__(self):
        self.messages = []
        self.message_queue = queue.Queue()

    def add_message(self, message: MessageUnit):
        self.messages.append(message)
        self.message_queue.put(message)

    def clear(self):
        self.messages.clear()
        self.message_queue = queue.Queue()

message_manager = MessageManager()