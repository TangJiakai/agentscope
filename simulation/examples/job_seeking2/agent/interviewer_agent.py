import os
import threading
from typing import Optional
from jinja2 import Environment, FileSystemLoader
from typing import Union
from typing import Sequence

from agentscope.agents import AgentBase
from agentscope.agents.agent import DistConf
from agentscope.message import Msg
from agentscope.models.response import ModelResponse
from agentscope.models import load_model_by_config_name

from simulation.examples.job_seeking.utils.utils import extract_dict
from simulation.helpers.message import MessageUnit, StateUnit, message_manager
from simulation.helpers.utils import setup_memory
from simulation.examples.job_seeking2.agent.seeker_agent import Seeker


scene_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_loader = FileSystemLoader(os.path.join(scene_path, "prompts"))
env = Environment(loader=file_loader)
Template = env.get_template("interviewer_prompts.j2").module


InterviewerAgentStates = [
    "idle", 
    "screening cv", 
    "making decision", 
    "external interviewing"
]

def set_state(flag: str):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            init_state = self._state
            self._state = flag
            try:
                return func(*args, **kwargs)
            finally:
                self._state = init_state
        return wrapper
    return decorator

class Job(object):
    def __init__(self, name: str, company_name: str, jd: str, jr: list, hc: int):
        self.name = name
        self.company_name = company_name
        self.jd = jd
        self.jr = jr
        self.hc = hc

    def __str__(self):
        jr_string = "\n".join([f"- {r}" for r in self.jr])
        return (
            f"Position Title: {self.name}\n"
            f"Company Name: {self.company_name}\n"
            f"Position Description: {self.jd}\n"
            f"Position Requirements:\n{jr_string}"
        )


class InterviewerAgent(AgentBase):
    """Interviewer agent."""

    name: str  # Name of the job
    model_config_name: str  # Model config name
    job: Job  # Job object
    system_prompt: Msg  # System prompt

    def __init__(
        self,
        name: str,
        model_config_name: str,
        company_name: str,
        jd: str,
        jr: list,
        hc: int,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            model_config_name=model_config_name,
            to_dist=(
                DistConf(host=kwargs["host"], port=kwargs["port"])
                if kwargs["distributed"]
                else None
            ),
        )
        self.model_config_name = model_config_name
        self.job = Job(name, company_name, jd, jr, hc)
        
        self._lock = threading.Lock()
        self._state = "idle"

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        state.pop("model")
        memory_state = self.memory.__dict__.copy()
        if "model" in memory_state:
            memory_state["model"] = None
        if "embedding_model" in memory_state:
            memory_state["embedding_model"] = None
        state["memory"] = memory_state
        return state

    def __setstate__(self, state: object) -> None:
        self.__dict__.update(state)
        self.memory = setup_memory(self.memory_config)
        self.memory.__dict__.update(state["memory"])
        self.model = load_model_by_config_name(self.model_config_name)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_value):
        if new_value not in InterviewerAgentStates:
            raise ValueError(f"Invalid state: {new_value}")
        self._state = new_value
        message_manager.add_state(StateUnit(agent_id=self.id, state=new_value))

    def set_id(self, id: int):
        self.id = id
        self.job.id = id

    def get_id(self):
        return self.id

    def set_embedding(self, embedding_model):
        self.embedding = embedding_model.encode(
            str(self.job), normalize_embeddings=True
        )

    def send_message(self, prompt, response):
        message_manager.add_message(MessageUnit(
            name=self.name,
            prompt="\n".join("\n".join([p["content"] for p in prompt])),
            completion=response.text,
            agent_type=type(self).__name__,
            agent_id=self.get_id(),
        ))

    def init_system_prompt(self, company):
        self.system_prompt = Msg(
            "system", Template.system_prompt(self.job, company), role="system"
        )

    def acquire_lock(self):
        self._lock.acquire()

    def release_lock(self):
        self._lock.release()

    @set_state("screening cv")
    def screening_cv_fun(self, seeker: Seeker):
        if self.job.hc <= 0:
            return Msg(self.name, "No headcount available", role="assistant")

        msg = Msg("user", Template.screening_cv_prompt(seeker, self.job.hc), role="user")
        return self.reply(msg)

    @set_state("making decision")
    def make_decision_fun(self, seeker: Seeker):
        msg = Msg("assistant", Template.make_decision_prompt(seeker), role="assistant")
        response = self.reply(msg)
        if "yes" in response.text:
            self.job.hc -= 1
        
        return response

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
        if fun := getattr(self, f"{x.fun}_fun"):
            fun(**getattr(x, "params", {}))
            return Msg(self.name, None, role="assistant")
        else:
            self.memory.add(x)
            memory = self.memory.get_memory(x)
            msg = Msg("assistant", "\n".join([p.content for p in memory]) + x.content, "assistant")
            prompt = self.model.format(self.system_prompt, msg)
            response = self.model(prompt)
            msg = Msg(self.name, response.text, role="assistant")
            self.send_message(prompt, response)
            self.memory.add(msg)
            
            return response
