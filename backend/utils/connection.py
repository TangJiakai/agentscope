from typing import Dict, Optional, Union

from fastapi import WebSocket

from simulation.helpers.message import MessageUnit


class ConnectionManager:
    def __init__(self):
        self.msg_connection: WebSocket
        self.agent_connections: Dict[int, WebSocket] = {}

    async def connect(self, websocket: WebSocket, agent_id: Optional[int] = None):
        await websocket.accept()
        if agent_id:
            self.agent_connections[agent_id] = websocket
        else:
            self.msg_connection = websocket

    async def disconnect(self, websocket: WebSocket, agent_id: Optional[int] = None):
        if agent_id:
            del self.agent_connections[agent_id]
        elif self.msg_connection == websocket:
            self.msg_connection = None
        await websocket.close()

    async def send(self, msg: Union[MessageUnit, str]):
        if isinstance(msg, MessageUnit):
            msg = msg.model_dump_json()
        await self.msg_connection.send_text(msg)

    async def send_to_agent(self, agent_id: int, msg: str):
        if agent_id not in self.agent_connections:
            return
        await self.agent_connections[agent_id].send_text(msg)

    def get_connection(self, agent_id: Optional[int] = None):
        if agent_id:
            return self.agent_connections.get(agent_id)
        return self.msg_connection


manager = ConnectionManager()
