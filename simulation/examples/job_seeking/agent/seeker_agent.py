import os
from typing import Optional
from jinja2 import Environment, FileSystemLoader
from typing import Union
from typing import Sequence

from agentscope.agents import AgentBase
from agentscope.agents.agent import DistConf
from agentscope.message import Msg
from agentscope.models import ModelResponse
from agentscope.models import load_model_by_config_name

from simulation.examples.job_seeking.utils.utils import extract_dict
from simulation.helpers.message import message_manager, MessageUnit
from simulation.helpers.utils import setup_memory


scene_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_loader = FileSystemLoader(os.path.join(scene_path, "prompts"))
env = Environment(loader=file_loader)
Template = env.get_template('seeker_prompts.j2').module


class Seeker(object):
    def __init__(self, name: str, cv: str, trait: str, status: str):
        self.name = name
        self.cv = cv
        self.trait = trait
        self.status = status
        self.job_condition = "unemployed"

    def __str__(self):
        return (
            f"Seeker Name: {self.name}\n"
            f"Seeker CV: {self.cv}\n"
        )


class SeekerAgent(AgentBase):
    """seeker agent."""

    name: str   # Name of the seeker
    model_config_name: str  # Model config name
    seeker: Seeker  # Seeker object
    system_prompt: Msg  # System prompt
    search_job_number: int  # Search job number
    job_ids_pool: list  # Job ids pool
    apply_job_ids: list  # Apply job ids
    cv_passed_job_ids: list  # CV passed job ids
    offer_job_ids: list  # Offer job ids
    wl_jobs_dict: dict  # Waitlist jobs dict
    decision: int  # Decision, 0: no any (wl) offer, 1: accept offer, 2: reject offer and wait jobs in waitlist, 3: reject offer and waitlist jobs, prepare for next round
    final_offer_id: int  # Final offer id
    reject_offer_job_ids: list  # Reject offer job ids
    reject_wl_job_ids: list  # Reject waitlist job ids
    fail_job_ids: list  # Fail job ids
    update_variables: list  # Update variables
    seeking: bool  # seeking job or not

    def __init__(
        self,
        name: str,
        model_config_name: str,
        cv: str,
        trait: str,
        status: str,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            model_config_name=model_config_name,
            to_dist=DistConf(host=kwargs["host"], port=kwargs["port"]) if kwargs["distributed"] else None
        )
        self.model_config_name = model_config_name
        self.seeker = Seeker(name, cv, trait, status)
        self.system_prompt = Msg("system", Template.system_prompt(self.seeker), role="system")
        self.memory_info = {
            "final_decision": 0,
            "waiting_time": 0,
        }
        self.job_ids_pool, self.apply_job_ids, self.cv_passed_job_ids, self.offer_job_ids, self.wl_jobs_dict = list(), list(), list(), list(), dict()
        self.fail_job_ids = list()
        self.update_variables = [self.job_ids_pool, self.apply_job_ids, self.offer_job_ids, self.wl_jobs_dict]

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
        self.memory = setup_memory(self.memory_config)
        self.memory.__dict__.update(state['memory'])
        self.model = load_model_by_config_name(self.model_config_name)

    def set_id(self, id: int):
        self.id = id
        self.seeker.id = id

    def get_id(self):
        return self.id
    
    def set_embedding(self, embedding_model):
        self.embedding = embedding_model.encode(str(self.seeker), normalize_embeddings=True)

    def search_job_number_fun(self):
        """Set search job number."""
        msg = Msg("user", Template.search_job_number_prompt(), role="user")
        tht = self.reflect(current_action="Determine the number of jobs to search for")
        prompt = self.model.format(self.system_prompt, tht, msg)

        def parse_func(response: ModelResponse) -> ModelResponse:
            message_manager.add_message(MessageUnit(
                name=self.name, 
                prompt='\n'.join([p['content'] for p in prompt]), 
                completion=response.text, 
                agent_type=type(self).__name__,
                agent_id=self.get_id()
            ))
            try:
                res_dict = extract_dict(response.text)
                return ModelResponse(raw=int(res_dict["number"]))
            except:
                raise ValueError(
                    f"Invalid response format in parse_func "
                    f"with response: {response.text}",
                )
        # print(prompt)
        response = self.model(prompt, parse_func=parse_func).raw
        # print(response)
        self.search_job_number = max(min(response, len(self.job_ids_pool)), 1)

    def apply_job_fun(self, search_jobs: list):
        """Apply job."""
        msg = Msg("user", Template.apply_jobs_prompt(search_jobs), role="user")
        tht = self.reflect(current_action="Select the positions to which you want to submit your resume")
        prompt = self.model.format(self.system_prompt, tht, msg)

        def parse_func(response: ModelResponse) -> ModelResponse:
            message_manager.add_message(MessageUnit(
                name=self.name, 
                prompt='\n'.join([p['content'] for p in prompt]), 
                completion=response.text, 
                agent_type=type(self).__name__,
                agent_id=self.get_id()
            ))
            try:
                res_dict = extract_dict(response.text)
                return ModelResponse(raw=list(map(int, res_dict["apply_jobs"])))
            except:
                raise ValueError(
                    f"Invalid response format in parse_func "
                    f"with response: {response.text}",
                )
        
        # print(prompt)
        response = self.model(prompt, parse_func=parse_func).raw
        # print(response)
        self.apply_job_ids = response

    def make_decision_fun(self, agents: dict):
        """Make decision."""
        if len(self.offer_job_ids) == 0 and len(self.wl_jobs_dict) == 0:
            self.decision = 0
            self.final_offer_id = None
            self.reject_offer_job_ids, self.reject_wl_job_ids = list(), list()

        msg = Msg("user", Template.make_decision_prompt(self.offer_job_ids, self.wl_jobs_dict, agents), role="user")
        tht = self.reflect(current_action="Decide to accept, wait for a backup, or decline the offer")
        prompt = self.model.format(self.system_prompt, tht, msg)

        def parse_func(response: ModelResponse) -> ModelResponse:
            message_manager.add_message(MessageUnit(
                name=self.name, 
                prompt='\n'.join([p['content'] for p in prompt]), 
                completion=response.text, 
                agent_type=type(self).__name__,
                agent_id=self.get_id()
            ))
            try:
                res_dict = extract_dict(response.text)
                res_dict = {k: int(v) if v else None for k, v in res_dict.items()}
                assert res_dict["decision"] in [1,2,3], ValueError(
                    f"Invalid response in parse_func "
                    f"with response: {response.text}",
                )
                if res_dict["decision"] == 1:   # Accept offer
                    final_offer_id = res_dict["final_offer_id"]
                    return ModelResponse(raw={
                        "decision": res_dict["decision"],
                        "final_offer_id": final_offer_id
                    })
                elif res_dict["decision"] in [2,3]:  # Reject offer or waitlist
                    return ModelResponse(raw={
                        "decision": res_dict["decision"]
                    })
            except:
                raise ValueError(
                    f"Invalid response format in parse_func "
                    f"with response: {response.text}",
                )
        
        # print(prompt)
        response = self.model(prompt, parse_func=parse_func).raw
        # print(response)
        if response["decision"] == 1:   # Accept offer
            self.decision = 1
            self.seeker.status = "employed"
            self.final_offer_id = response["final_offer_id"]
            self.reject_offer_job_ids = list(set(self.offer_job_ids) - set([self.final_offer_id]))
            self.reject_wl_job_ids = [x for x in self.wl_jobs_dict]
        elif response["decision"] == 2: # Reject offer and wait jobs in waitlist
            self.decision = 2
            self.final_offer_id = None
            self.reject_offer_job_ids = self.offer_job_ids
            self.reject_wl_job_ids = list()
        elif response["decision"] == 3: # Reject offer and waitlist jobs, prepare for next round
            self.decision = 3
            self.final_offer_id = None
            self.reject_wl_job_ids = [x for x in self.wl_jobs_dict]
            self.wl_jobs_dict = dict()
            self.reject_offer_job_ids = self.offer_job_ids
        
        self.offer_job_ids = list()

    def add_memory_fun(self, seeking=True):
        if seeking:
            mem = Msg("assistant", Template.seeker_memory(self.memory_info), role="assistant")
        else: # agents who are on the job and do not seek jobs
            mem = Msg("assistant", Template.nonseeker_memory(self.seeker), role="assistant")
        self.memory.add(mem)

    def reflect(self, current_action):
        """Reflect from memories."""
        query_msg = Msg("assistant", current_action, role="assistant")
        retrived_memories = self.memory.get_memory(query_msg)
        if retrived_memories is None or len(retrived_memories) == 0:
            return Msg("assistant", None, role="assistant")
        msg = Msg("user", Template.reflection_prompt([x["content"] for x in retrived_memories], current_action), role="user")
        prompt = self.model.format(self.system_prompt, msg)

        response = self.model(prompt).text
        return Msg("user", content=response, role="user")
    
    def determine_status_fun(self):
        msg = Msg("user", Template.determine_status_prompt(), role="user")
        tht = self.reflect(current_action="Choose whether to conduct a job search")
        prompt = self.model.format(self.system_prompt, tht, msg)

        def parse_func(response: ModelResponse) -> ModelResponse:
            message_manager.add_message(MessageUnit(
                name=self.name, 
                prompt='\n'.join([p['content'] for p in prompt]), 
                completion=response.text, 
                agent_type=type(self).__name__,
                agent_id=self.get_id()
            ))
            try:
                res_dict = response.text.strip().lower()
                if 'yes' in res_dict:
                    return ModelResponse(raw=True)
                return ModelResponse(raw=False)
            except:
                raise ValueError(
                    f"Invalid response format in parse_func "
                    f"with response: {response.text}",
                )
        
        response = self.model(prompt, parse_func=parse_func).raw
        self.seeking = response
    
    def update_job_condition(self,company_name, job):
        self.seeker.job_condition = Template.generate_job_condition(company_name,job)
        
    def update_fun(self):
        self.memory_info = {
            "final_decision": 0,
            "waiting_time": 0,
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