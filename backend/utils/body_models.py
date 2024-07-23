from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


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
    args: Dict


class AgentConfig(BaseModel):
    cls: str = Field(..., alias="class")
    num_agents: int


class CheckpointResp(BaseModel):
    run_name: str
    pkls: List[str]


class CheckpointReq(BaseModel):
    path: str


class FilterCondition(BaseModel):
    type: Literal["None", "turn", "id", "name"]
    turns: Optional[List[int]] = None
    ids: Optional[List[int]] = None
    names: Optional[List[str]] = None
