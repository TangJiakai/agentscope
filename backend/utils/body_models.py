from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field
from pydantic.json_schema import GenerateJsonSchema
from pydantic.config import ConfigDict

from simulation.helpers.message import MessageUnit


class MessagesResp(BaseModel):
    msgs: List[MessageUnit]
    cur_page: int
    total_pages: int


class Scene(BaseModel):
    name: str
    desc: str
    pic_path: str


class ClientArgs(BaseModel):
    model_config = ConfigDict(title="Client Args", description="The client argument configuration.")
    max_retries: int = Field(10, ge=0, description="The maximum number of retries to make on a request.")
    base_url: str = Field("http://localhost:8084/v1", description="The base URL for the model.")
    timeout: float = Field(6000, ge=0, description="The timeout for a request.")


class GenerateArgs(BaseModel):
    model_config = ConfigDict(title="Generate Args", description="The LLM generate argument configuration.")
    temperature: float = Field(0, ge=0, le=1, description="The temperature for sampling.")
    max_tokens: int = Field(512, ge=1, description="The maximum number of tokens to generate.")


class ModelConfig(BaseModel):
    model_config = ConfigDict(title="Model Config", description="The LLM model configuration.")
    config_name: str = Field("LLM", description="The name of the model used for associating with agents.")
    model_type: str = Field("openai_chat", description="The type of the model, e.g., openai_chat.")
    model_name: str = Field("/data/pretrain_dir/Meta-Llama-3-8B-Instruct", description="Your LLM model path or id set in vllm.")
    api_key: str = Field("api_key", description="The API key for the model.")
    client_args: ClientArgs = ClientArgs()
    generate_args: GenerateArgs = None


class MemoryConfig(BaseModel):
    cls: str = Field(..., alias="class")
    args: Optional[Dict] = None


class NoneMemoryArgs(BaseModel):
    model_config = ConfigDict(title="None Memory Arguments", description="The memory arguments for NoneMemory.")


class NoneMemoryConfig(BaseModel):
    model_config = ConfigDict(title="None Memory Config", description="The memory configuration for NoneMemory.")
    cls: str = Field("NoneMemory", alias="class")
    args: NoneMemoryArgs


class ShortMemoryArgs(BaseModel):
    model_config = ConfigDict(title="Short Memory Arguments", description="The memory arguments for ShortMemory.")
    stm_K: int = Field(2, ge=1, description="The number of short-term memory to keep.")


class ShortMemoryConfig(BaseModel):
    model_config = ConfigDict(title="Short Memory Config", description="The memory configuration for ShortMemory.")
    cls: str = Field("ShortMemory", alias="class")
    args: ShortMemoryArgs


class ShortLongMemoryArgs(BaseModel):
    model_config = ConfigDict(title="ShortLong Memory Arguments", description="The memory arguments for ShortLongMemory.")
    importance_weight: float = Field(0.15, ge=0, le=1, description="The importance weight used for computing memory importance score.")
    stm_K: int = Field(2, ge=1, description="The number of short-term memory to keep.")
    ltm_K: int = Field(2, ge=1, description="The number of long-term memory to keep.")


class ShortLongMemoryConfig(BaseModel):
    model_config = ConfigDict(title="ShortLong Memory Config", description="The memory configuration for ShortLongMemory.")
    cls: str = Field("ShortLongMemory", alias="class")
    args: ShortLongMemoryArgs


class ShortLongReflectionMemoryArgs(BaseModel):
    model_config = ConfigDict(title="ShortLongReflection Memory Arguments", description="The memory arguments for ShortLongReflectionMemory.")
    importance_weight: float = Field(0.15, ge=0, le=1, description="The importance weight used for computing memory importance score.")
    stm_K: int = Field(2, ge=1, description="The number of short-term memory to keep.")
    ltm_K: int = Field(2, ge=1, description="The number of long-term memory to keep.")
    reflection_threshold: float = Field(1, ge=0, le=1, description="The reflection threshold used to control whether to reflect.")


class ShortLongReflectionMemoryConfig(BaseModel):
    model_config = ConfigDict(title="ShortLongReflection Memory Config", description="The memory configuration for ShortLongReflectionMemory.")
    cls: str = Field("ShortLongReflectionMemory", alias="class")
    args: ShortLongReflectionMemoryArgs


class AllMemoryConfig(BaseModel):
    model_config = ConfigDict(title="All Memory Config", description="The memory configuration for all memory.")
    memory_config: Union[NoneMemoryConfig, ShortMemoryConfig, ShortLongMemoryConfig, ShortLongReflectionMemoryConfig]


class MemoryGenerateJsonSchema(GenerateJsonSchema):
    def generate(self, schema, mode='validation'):
        json_schema = super().generate(schema, mode=mode)
        del json_schema['title']
        del json_schema['required']
        json_schema['anyOf'] = json_schema['properties']['memory_config']['anyOf']
        del json_schema['properties']
        return json_schema


class AgentConfig(BaseModel):
    cls: str = Field(..., alias="class")
    num_agents: int


class AgentProfileConfig(BaseModel):
    cls: str = Field(..., alias="class")
    cur_num_agents: int
    max_num_agents: int


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
    distributed: bool = True
    args: Optional[DistributedArgs] = None


class FilterCondition(BaseModel):
    condition: Literal["None", "id", "name", "type"]
    ids: Optional[List[int]] = None
    names: Optional[List[str]] = None
    types: Optional[List[str]] = None


class BroadcastMsg(BaseModel):
    msg: str


class Coord(BaseModel):
    x: float
    y: float


class AgentInfo(BaseModel):
    name: str
    id: str
    cls: str
    state: str
    profile: str
    gender: Optional[Literal["female", "male"]] = None
    coordinates: Coord
    avatar: str


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
    rewritten_response: Optional[str] = ""
    rating: Optional[int] = 0


class Transform(BaseModel):
    scaleX: float = 1
    scaleY: float = 1
    translateX: float = 0
    translateY: float = 0
    skewX: float = 0
    skewY: float = 0