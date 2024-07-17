from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class _ClientArgs(BaseModel):
    base_url: str
    max_reties: int


class _GenerateArgs(BaseModel):
    temperature: float = Field(ge=0, le=1)
    max_tokens: int


class ModelConfig(BaseModel):
    config_name: str
    model_type: str
    model_name: str
    api_key: str
    client_args: _ClientArgs
    generate_args: Optional[_GenerateArgs]


class ModelConfigs(BaseModel):
    model_configs: List[ModelConfig]


class MemoryConfig(BaseModel):
    cls: str = Field(..., alias="class")
    args: Dict


class CheckpointResp(BaseModel):
    run_name: str
    pkls: List[str]


class CheckpointReq(BaseModel):
    run_name: str
    pkl: str


class FilterCondition(BaseModel):
    type: Literal["None", "turn", "id", "name"]
    turns: Optional[List[int]]
    ids: Optional[List[int]]
    names: Optional[List[str]]
