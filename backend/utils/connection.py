from typing import Dict, Optional, Union

from fastapi import WebSocket

from simulation.helpers.message import StateUnit


class ConnectionManager:
    def __init__(self):
        self.state_connection: WebSocket
        self.agent_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, agent_id: Optional[str] = None):
        await websocket.accept()
        if agent_id:
            self.agent_connections[agent_id] = websocket
        else:
            self.state_connection = websocket

    async def disconnect(self, websocket: WebSocket, agent_id: Optional[str] = None):
        if agent_id:
            del self.agent_connections[agent_id]
        elif self.state_connection == websocket:
            self.state_connection = None
        await websocket.close()

    async def send(self, state: Union[StateUnit, str]):
        if isinstance(state, StateUnit):
            state = state.model_dump_json()
        await self.state_connection.send_text(state)

    async def send_to_agent(self, agent_id: str, msg: str):
        if agent_id not in self.agent_connections:
            return
        await self.agent_connections[agent_id].send_text(msg)

    def get_connection(self, agent_id: Optional[str] = None):
        if agent_id:
            return self.agent_connections.get(agent_id)
        return self.state_connection


manager = ConnectionManager()
