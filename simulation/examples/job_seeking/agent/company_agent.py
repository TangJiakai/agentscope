import os
from typing import Optional
from jinja2 import Environment, FileSystemLoader
from typing import Union
from typing import Sequence

from agentscope.agents import AgentBase
from agentscope.agents.agent import DistConf
from agentscope.message import Msg
from agentscope.models import load_model_by_config_name

from simulation.helpers.message import MessageUnit, StateUnit, message_manager
from simulation.helpers.utils import setup_memory

scene_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_loader = FileSystemLoader(os.path.join(scene_path, "prompts"))
env = Environment(loader=file_loader)
Template = env.get_template('company_prompts.j2').module


CompanyAgentStates = [
    "idle",
    "external interviewing",
]


class Company(object):
    """company object."""
    def __init__(self, name: str, cd: str):
        self.name = name
        self.cd = cd

    def __str__(self):
        return (
            f"Company Name: {self.name}\n"
            f"Company Description: {self.cd}"
        )
    

def set_state(flag: str):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            init_state = self._state
            self._state = flag
            try:
                return func(self, *args, **kwargs)
            finally:
                self._state = init_state
        return wrapper
    return decorator


class CompanyAgent(AgentBase):
    """company agent."""

    name: str   # Name of the company
    model_config_name: str  # Model config name
    company: Company  # Company object
    system_prompt: Msg  # System prompt

    def __init__(
        self,
        name: str,
        model_config_name: str,
        cd: str,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            model_config_name=model_config_name,
            to_dist=DistConf(host=kwargs["host"], port=kwargs["port"]) if kwargs["distributed"] else None
        )
        self.model_config_name = model_config_name
        self.company = Company(name, cd)
        self.system_prompt = Msg("system", Template.system_prompt(self.company), role="system")
        self._state = "idle"

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        state.pop("model")
        memory_state = self.memory.__dict__.copy()
        if "model" in memory_state:
            memory_state["model"] = None
        if "embedding_model" in memory_state:
            memory_state["embedding_model"] = None
        state['memory'] = memory_state
        return state
    
    def __setstate__(self, state: object) -> None:
        self.__dict__.update(state)
        self.memory = setup_memory(self.memory_config)
        self.memory.__dict__.update(state['memory'])
        self.model = load_model_by_config_name(self.model_config_name)
    
    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_value):
        if new_value not in CompanyAgentStates:
            raise ValueError(f"Invalid state: {new_value}")
        self._state = new_value
        message_manager.add_state(StateUnit(agent_id=self.id, state=new_value))

    def set_id(self, id: int):
        self.id = id
        self.company.id = id

    def get_id(self):
        return self.id
    
    def send_message(self, prompt, response):
        message_manager.add_message(MessageUnit(
            name=self.name,
            query="\n".join("\n".join([p["content"] for p in prompt])),
            response=response.text,
            agent_type=type(self).__name__,
            agent_id=self.get_id(),
        ))

    @set_state("external interviewing")
    def external_interview_fun(self, query, **kwargs):
        query_msg = Msg("assistant", query, role="assistant")
        memory_msg = self.memory.get_memory(query_msg)
        msg = Msg("assistant", "\n".join([p.content for p in memory_msg]) + query, "assistant")
        prompt = self.model.format(self.system_prompt, msg)
        response = self.model(prompt)
        return response.text
    
    def run_fun(self, **kwargs):
        pass
    
    def reply(self, x: Optional[Union[Msg, Sequence[Msg]]] = None) -> Msg:
        if x is not None and x.get("fun", None):
            return getattr(self, f"{x.fun}_fun")(**getattr(x, "params", {}))
        else:
            memory = self.memory.get_memory(x)
            self.memory.add(x)
            msg = Msg("user", "\n".join([p.content for p in memory]) + x.content, "user")
            prompt = self.model.format(self.system_prompt, msg)
            response = self.model(prompt)
            self.send_message(prompt, response)
            msg = Msg(self.name, response.text, role="user")
            self.memory.add(msg)
            return msg