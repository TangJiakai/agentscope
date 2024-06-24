from typing import Optional
from jinja2 import Environment, FileSystemLoader
import json

from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.models import ModelResponse

from utils.utils import extract_json_string

file_loader = FileSystemLoader("prompts")
env = Environment(loader=file_loader)
Template = env.get_template('seeker_prompts.j2').module


class Seeker(object):
    def __init__(self, id: int, name: str, cv: str, trait: str, status: str):
        self.id = id
        self.name = name
        self.cv = cv
        self.trait = trait
        self.status = status


class SeekerAgent(AgentBase):
    """seeker agent."""

    name: str   # Name of the seeker
    model_config_name: str  # Model config name
    seeker: Seeker  # Seeker object
    system_prompt: Msg  # System prompt
    search_job_number: int  # Search job number
    job_ids_pool: list  # Job ids pool
    apply_job_ids: list  # Apply job ids
    offer_job_ids: list  # Offer job ids
    wl_jobs_dict: dict  # Waitlist jobs dict
    decision: int  # Decision, 0: no any (wl) offer, 1: accept offer, 2: reject offer and wait jobs in waitlist, 3: reject offer and waitlist jobs, prepare for next round
    final_offer_id: int  # Final offer id
    reject_offer_job_ids: list  # Reject offer job ids
    reject_wl_job_ids: list  # Reject waitlist job ids
    update_variables: list  # Update variables

    def __init__(
        self,
        name: str,
        model_config_name: str,
        id: int,
        cv: str,
        trait: str,
        status: str,
    ) -> None:
        super().__init__(
            name=name,
            model_config_name=model_config_name,
        )
        self.seeker = Seeker(id, name, cv, trait, status)
        self.system_prompt = Msg("system", Template.system_prompt(self.seeker), role="system")
        self.memory_info = {
            "final_decision": 0,
            "waiting_time": 0,
        }

        self.job_ids_pool, self.apply_job_ids, self.offer_job_ids, self.wl_jobs_dict = list(), list(), list(), dict()
        self.update_variables = [self.job_ids_pool, self.apply_job_ids, self.offer_job_ids, self.wl_jobs_dict]
    
    def get_id(self):
        """Return the id of the seeker."""
        return self.seeker.id

    def search_job_number_fun(self):
        """Set search job number."""
        msg = Msg("user", Template.search_job_number_prompt(), role="user")
        prompt = self.model.format(self.system_prompt, self.memory.get_memory(self.recent_n), msg)

        def parse_func(response: ModelResponse) -> ModelResponse:
            try:
                res_dict = json.loads(extract_json_string(response.text))
                return ModelResponse(raw=int(res_dict["number"]))
            except:
                raise ValueError(
                    f"Invalid response format in parse_func "
                    f"with response: {response.text}",
                )
        # print(prompt)
        response = self.model(prompt, parse_func=parse_func).raw
        # print(response)
        self.search_job_number = min(response, len(self.job_ids_pool))

    def apply_job_fun(self, search_jobs: list):
        """Apply job."""
        msg = Msg("user", Template.apply_jobs_prompt(search_jobs), role="user")
        prompt = self.model.format(self.system_prompt, self.memory.get_memory(self.recent_n), msg)

        def parse_func(response: ModelResponse) -> ModelResponse:
            try:
                res_dict = json.loads(extract_json_string(response.text))
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

    def make_decision_fun(self, id2job: dict):
        """Make decision."""
        if len(self.offer_job_ids) == 0 and len(self.wl_jobs_dict) == 0:
            self.decision = 0
            self.final_offer_id = None
            self.reject_offer_job_ids, self.reject_wl_job_ids = list(), list()

        if len(self.offer_job_ids) > 0:
            print(f"总计得到{len(self.offer_job_ids)}个Offer，分别如下:\n")
            for job_id in self.offer_job_ids:
                print(job_id, id2job[job_id]["agent"].job.name, id2job[job_id]["agent"].job.jd, id2job[job_id]["agent"].job.company)

        if len(self.wl_jobs_dict) > 0:
            print(f"总计得到{len(self.wl_jobs_dict)}个递补Offer，分别如下:\n")
            for wl_job_id, wl_job in self.wl_jobs_dict.items():
                print(wl_job_id, id2job[wl_job_id]["agent"].job.name, id2job[wl_job_id]["agent"].job.jd, id2job[wl_job_id]["agent"].job.company, wl_job.rank, wl_job.wl_n)


        msg = Msg("user", Template.make_decision_prompt(self.offer_job_ids, self.wl_jobs_dict, id2job), role="user")
        prompt = self.model.format(self.system_prompt, self.memory.get_memory(self.recent_n), msg)

        def parse_func(response: ModelResponse) -> ModelResponse:
            try:
                res_dict = json.loads(extract_json_string(response.text))
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
        
        print(prompt)
        response = self.model(prompt, parse_func=parse_func).raw
        print(response)
        if response["decision"] == 1:   # Accept offer
            self.decision = 1
            self.seeker.status = "在职"
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
            self.wl_jobs_dict = dict()
            self.reject_offer_job_ids = self.offer_job_ids
            self.reject_wl_job_ids = [x for x in self.wl_jobs_dict]
        
        self.offer_job_ids = list()

    def add_memory(self):
        mem = Msg("assistant", Template.seeker_memory(self.memory_info), role="assistant")
        print(mem)
        self.memory.add(mem)

    def update_fun(self):
        for var in self.update_variables:
            var.clear()

    def reply(self, x: Optional[dict] = None) -> dict:
        return Msg(self.name, None, role="assistant")