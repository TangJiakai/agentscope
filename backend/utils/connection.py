from typing import Dict, List, Literal, Optional, Union

from fastapi import WebSocket
from pydantic import BaseModel

from simulation.helpers.message import StateUnit


class ChatMessage(BaseModel):
    sender: Literal["human", "agent"]
    message: str


class ConnectionManager:
    def __init__(self):
        self.state_connection: WebSocket = None
        self.all_agents_state: Dict[str, str] = {}
        self.agent_connections: Dict[str, WebSocket] = {}
        self.agent_connections_history: Dict[str, List[ChatMessage]] = {}

    async def connect(self, websocket: WebSocket, agent_id: Optional[str] = None):
        await websocket.accept()
        if agent_id:
            self.agent_connections[agent_id] = websocket
            self.agent_connections_history[agent_id] = []
        else:
            self.state_connection = websocket
            await self.state_connection.send_text(str(self.all_agents_state))

    async def disconnect(self, websocket: WebSocket, agent_id: Optional[str] = None):
        if agent_id:
            del self.agent_connections[agent_id]
            del self.agent_connections_history[agent_id]
        elif self.state_connection == websocket:
            self.state_connection = None
        await websocket.close()

    async def send(self, state: Union[StateUnit, str]):
        if isinstance(state, StateUnit):
            self.all_agents_state[state.agent_id] = state.state
            # state = state.model_dump_json()
        await self.state_connection.send_text(str(self.all_agents_state))

    async def send_to_agent(self, agent_id: str, msg: str):
        if agent_id not in self.agent_connections:
            return
        self.agent_connections_history[agent_id].append(ChatMessage(sender="agent", message=msg))
        send_msgs = [msg.model_dump_json() for msg in self.agent_connections_history[agent_id]]
        await self.agent_connections[agent_id].send_text(f"[{', '.join(send_msgs)}]")

    def get_connection(self, agent_id: Optional[str] = None):
        if agent_id:
            return self.agent_connections.get(agent_id)
        return self.state_connection

    def clear(self):
        if self.state_connection:
            self.state_connection.close()
        for connection in self.agent_connections.values():
            connection.close()
        self.state_connection = None
        self.all_agents_state.clear()
        self.agent_connections.clear()
        self.agent_connections_history.clear()


manager = ConnectionManager()
