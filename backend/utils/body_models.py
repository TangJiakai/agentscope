from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Scene(BaseModel):
    name: str
    desc: str
    pic_path: str


class _ClientArgs(BaseModel):
    max_retries: int
    base_url: str


class _GenerateArgs(BaseModel):
    temperature: float = Field(ge=0, le=1)
    max_tokens: Optional[int] = None


class ModelConfig(BaseModel):
    config_name: str
    model_type: str
    model_name: str
    api_key: str
    client_args: _ClientArgs
    generate_args: Optional[_GenerateArgs] = None


class MemoryConfig(BaseModel):
    cls: str = Field(..., alias="class")
    args: Optional[Dict] = None


class AgentConfig(BaseModel):
    cls: str = Field(..., alias="class")
    num_agents: int


class CheckpointResp(BaseModel):
    run_name: str
    pkls: List[str]


class PathReq(BaseModel):
    path: Optional[str] = None


class DistributedArgs(BaseModel):
    host: Optional[str] = "localhost"
    base_port: int = 12200
    server_num_per_host: int = 10


class DistributedConfig(BaseModel):
    distributed: bool = False
    args: Optional[DistributedArgs] = None


class FilterCondition(BaseModel):
    condition: Literal["None", "id", "name", "type"]
    ids: Optional[List[int]] = None
    names: Optional[List[str]] = None
    types: Optional[List[str]] = None


class InterventionMsg(BaseModel):
    msg: str


class AgentInfo(BaseModel):
    name: str
    id: str
    cls: str
    state: str
    profile: str


class AgentStateInfo(BaseModel):
    agent_cls_name: str
    states: List[str]


class AgentState(BaseModel):
    agent_id: str
    state: str


class GPTReq(BaseModel):
    msg_ids: List[int]
    mode: Literal["rewrite", "rate"]


class ChangedMsg(BaseModel):
    msg_id: int
    rewritten_response: Optional[str] = None
    rating: Optional[int] = None