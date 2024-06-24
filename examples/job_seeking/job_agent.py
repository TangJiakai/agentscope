from typing import Optional
from jinja2 import Environment, FileSystemLoader

from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.models.response import ModelResponse

from utils.utils import extract_dict

file_loader = FileSystemLoader("prompts")
env = Environment(loader=file_loader)
Template = env.get_template('job_prompts.j2').module


class Job(object):
    def __init__(self, id: int, company_id: int, name: str, jd: str, jr: list, hc: int):
        self.id = id
        self.company_id = company_id
        self.name = name
        self.jd = jd
        self.jr = jr
        self.hc = hc

    def __str__(self):
        jr_string = "\n".join([f"- {r}" for r in self.jr])
        return (
            f"Position Title: {self.name}\n"
            f"Position Description: {self.jd}\n"
            f"Position Requirements:\n{jr_string}"
        )
    

class JobAgent(AgentBase):
    """job agent."""

    name: str   # Name of the job
    model_config_name: str  # Model config name
    job: Job  # Job object
    system_prompt: Msg  # System prompt
    hc: int  # Headcount
    cv_passed_seeker_ids: list  # CV passed seeker ids
    offer_seeker_ids: list  # Offer seeker ids
    wl_seeker_ids: list  # Waitlist seeker ids
    reject_seeker_ids: list  # Reject seeker ids
    update_variables: list  # Update variables

    def __init__(
        self,
        name: str,
        model_config_name: str,
        id: int,
        company_id: int,
        jd: str,
        jr: list,
        hc: int,
    ) -> None:
        super().__init__(
            name=name,
            model_config_name=model_config_name
        )
        self.job = Job(id, company_id, name, jd, jr, hc)
        self.hc = hc

        self.cv_passed_seeker_ids, self.offer_seeker_ids, self.wl_seeker_ids, self.reject_seeker_ids = list(), list(), list(), list()
        self.update_variables = [self.cv_passed_seeker_ids, self.offer_seeker_ids, self.wl_seeker_ids, self.reject_seeker_ids]

    def get_id(self):
        return self.job.id

    def init_system_prompt(self, company):
        self.system_prompt = Msg("system", Template.system_prompt(self.job, company), role="system")

    def cv_screening_fun(self, apply_seekers: list, excess_cv_passed_n: int):
        cv_passed_hc = min(self.hc+excess_cv_passed_n, len(apply_seekers))
        msg = Msg("user", Template.screen_resumes(cv_passed_hc, apply_seekers), role="user")
        prompt = self.model.format(self.system_prompt, self.memory.get_memory(self.recent_n), msg)

        def parse_func(response: ModelResponse) -> ModelResponse:
            try:
                res_dict = extract_dict(response.text)
                return ModelResponse(raw=list(map(int, res_dict["cv_passed_seeker_ids"])))
            except:
                raise ValueError(
                    f"Invalid response format in parse_func "
                    f"with response: {response.text}",
                )

        print(prompt)
        response = self.model(prompt, parse_func=parse_func).raw
        print(response)
        self.cv_passed_seeker_ids = response

    def interview_fun(self, cv_passed_seekers: list):
        pass

    def make_decision_fun(self, interview_seekers: list, wl_n: int):
        offer_hc = min(self.hc, len(interview_seekers))
        wl_n = min(wl_n, len(interview_seekers) - offer_hc)
        msg = Msg("user", Template.make_decision(offer_hc, wl_n, interview_seekers), role="user")
        prompt = self.model.format(self.system_prompt, self.memory.get_memory(self.recent_n), msg)

        def parse_func(response: ModelResponse) -> ModelResponse:
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
            
        print(prompt)
        response = self.model(prompt, parse_func=parse_func).raw
        print(response)
        self.offer_seeker_ids = response["offer_seeker_ids"]
        self.wl_seeker_ids = response["wl_seeker_ids"]
        self.reject_seeker_ids = list(set([seeker.id for seeker in interview_seekers]) - set(self.offer_seeker_ids) - set(self.wl_seeker_ids))

    def add_memory(self):
        pass

    def update_fun(self):
        for var in self.update_variables:
            var.clear()

    def reply(self, x: Optional[dict] = None) -> dict:
        return Msg(self.name, None, role="assistant")