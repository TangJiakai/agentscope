from typing import Optional
from jinja2 import Environment, FileSystemLoader
from typing import Union
from typing import Sequence

from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.models.response import ModelResponse
from agentscope.models import load_model_by_config_name

from utils.utils import extract_dict
from simulation.helpers.message import MessageUnit, message_manager
from simulation.helpers.utils import setup_memory


file_loader = FileSystemLoader("prompts")
env = Environment(loader=file_loader)
Template = env.get_template('job_prompts.j2').module


class Job(object):
    def __init__(self, name: str, company_name: str, jd: str, jr: list, hc: int):
        self.name = name
        self.company_name = company_name
        self.jd = jd
        self.jr = jr
        self.hc = hc
        self.emb = None

    def __str__(self):
        jr_string = "\n".join([f"- {r}" for r in self.jr])
        return (
            f"Position Title: {self.name}\n"
            f"Company Name: {self.company_name}\n"
            f"Position Description: {self.jd}\n"
            f"Position Requirements:\n{jr_string}"
        )
    

class JobAgent(AgentBase):
    """job agent."""

    name: str   # Name of the job
    model_config_name: str  # Model config name
    job: Job  # Job object
    system_prompt: Msg  # System prompt
    cv_passed_seeker_ids: list  # CV passed seeker ids
    offer_seeker_ids: list  # Offer seeker ids
    wl_seeker_ids: list  # Waitlist seeker ids
    reject_seeker_ids: list  # Reject seeker ids
    apply_seeker_ids: list  # Apply seeker ids
    update_variables: list  # Update variables

    def __init__(
        self,
        name: str,
        model_config_name: str,
        company_name: str,
        jd: str,
        jr: list,
        hc: int,
        memory_config: dict,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            model_config_name=model_config_name
        )
        self.model_config_name = model_config_name
        self.memory_config = memory_config
        self.memory = setup_memory(memory_config)
        self.job = Job(name, company_name, jd, jr, hc)
        self.memory_info = {
            "final_offer_seeker":[],
        }

        self.cv_passed_seeker_ids, self.offer_seeker_ids, self.wl_seeker_ids, self.reject_seeker_ids = list(), list(), list(), list()
        self.apply_seeker_ids = list()
        self.update_variables = [self.cv_passed_seeker_ids, self.offer_seeker_ids, self.wl_seeker_ids, self.reject_seeker_ids, self.apply_seeker_ids]

    def set_id(self, id: int):
        self.id = id

    def get_id(self):
        return self.id
    
    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        state.pop("model")
        memory_state = self.memory.__dict__.copy()
        try:
            memory_state["model"] = None
            memory_state["embedding_model"] = None
        except:
            pass
        state['memory'] = memory_state
        return state

    def __setstate__(self, state: object) -> None:
        self.__dict__.update(state)
        self.model = load_model_by_config_name(self.model_config_name)
        if getattr(self.model, "model"):
            self.memory.model = load_model_by_config_name(self.memory_config["model_config_name"])
        if getattr(self.model, "embedding_model"):
            self.memory.embedding_model = load_model_by_config_name(self.memory_config["embedding_model_config_name"])

    def init_system_prompt(self, company):
        self.system_prompt = Msg("system", Template.system_prompt(self.job, company), role="system")

    def cv_screening_fun(self, apply_seekers: list, excess_cv_passed_n: int):
        cv_passed_hc = min(self.job.hc+excess_cv_passed_n, len(apply_seekers))
        if cv_passed_hc == 0:
            self.cv_passed_seeker_ids = list()
            return
        
        msg = Msg("user", Template.screen_resumes(cv_passed_hc, apply_seekers), role="user")
        prompt = self.model.format(self.system_prompt, self.memory.get_memory(self.recent_n), msg)

        def parse_func(response: ModelResponse) -> ModelResponse:
            from simulator import CUR_TURN
            message_manager.add_message(MessageUnit(
                round=CUR_TURN, 
                name=self.name, 
                prompt='\n'.join([p['content'] for p in prompt]), 
                completion=response.text, 
                agent_type=type(self).__name__,
                agent_id=self.get_id()
            ))
            try:
                res_dict = extract_dict(response.text)
                return ModelResponse(raw=list(map(int, res_dict["cv_passed_seeker_ids"])))
            except:
                raise ValueError(
                    f"Invalid response format in parse_func "
                    f"with response: {response.text}",
                )

        response = self.model(prompt, parse_func=parse_func).raw
        self.cv_passed_seeker_ids = response

    def make_decision_fun(self, interview_seekers: list, wl_n: int):
        offer_hc = min(self.job.hc, len(interview_seekers))
        wl_n = min(wl_n, len(interview_seekers) - offer_hc)
        if offer_hc == 0:
            self.offer_seeker_ids = list()
            self.wl_seeker_ids = list()
            self.reject_seeker_ids = [seeker.id for seeker in interview_seekers]
            return
        
        msg = Msg("user", Template.make_decision(offer_hc, wl_n, interview_seekers), role="user")
        prompt = self.model.format(self.system_prompt, self.memory.get_memory(self.recent_n), msg)

        def parse_func(response: ModelResponse) -> ModelResponse:
            from simulator import CUR_TURN
            message_manager.add_message(MessageUnit(
                round=CUR_TURN, 
                name=self.name, 
                prompt='\n'.join([p['content'] for p in prompt]), 
                completion=response.text, 
                agent_type=type(self).__name__,
                agent_id=self.get_id()
            ))
            try:
                res_dict = extract_dict(response.text)
                return ModelResponse(raw={
                    "offer_seeker_ids": list(map(int, res_dict["offer_seeker_ids"])),
                    "wl_seeker_ids": list(map(int, res_dict["wl_seeker_ids"]))
                })
            except:
                raise ValueError(
                    f"Invalid response format in parse_func "
                    f"with response: {response.text}",
                )
            
        response = self.model(prompt, parse_func=parse_func).raw
        self.offer_seeker_ids = response["offer_seeker_ids"]
        self.wl_seeker_ids = response["wl_seeker_ids"]
        self.reject_seeker_ids = list(set([seeker.id for seeker in interview_seekers]) - set(self.offer_seeker_ids) - set(self.wl_seeker_ids))

    def add_memory_fun(self):
        mem = Msg("assistant", Template.job_memory(self.memory_info), role="assistant")
        self.memory.add(mem)

    def update_fun(self):
        self.memory_info = {
            "final_offer_seeker":[],
        }
        for var in self.update_variables:
            var.clear()

    def reply(self, x: Optional[Union[Msg, Sequence[Msg]]] = None) -> Msg:
        fun = getattr(self, f"{x.fun}_fun")
        if hasattr(x, "params"):
            fun(**x.params)
        else:
            fun()
        return Msg(self.name, None, role="assistant")