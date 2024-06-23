from typing import Optional
from jinja2 import Environment, FileSystemLoader
import json

from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.models.response import ModelResponse

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

    def get_id(self):
        return self.job.id

    def init_system_prompt(self, company):
        self.system_prompt = Msg("system", Template.system_prompt(self.job, company), role="system")

    def cv_screening_fun(self, apply_seekers: list, excess_cv_passed_n: int):
        cv_passed_hc = min(self.hc+excess_cv_passed_n, len(apply_seekers))
        msg = Msg("user", Template.screen_resumes(cv_passed_hc, apply_seekers), role="user")
        prompt = self.model.format(self.system_prompt, self.memory.get_memory(self.recent_n), msg)

        def parse_func(response: ModelResponse) -> ModelResponse:
            res_dict = json.loads(response.text)
            if "cv_passed_seekers" in res_dict:
                return ModelResponse(raw=list(map(int, res_dict["cv_passed_seekers"])))
            else:
                raise ValueError(
                    f"Invalid response format in parse_func "
                    f"with response: {response.text}",
                )

        # print(prompt)
        response = self.model(prompt, parse_func=parse_func).raw
        # print(response)
        self.cv_passed_seekers = response

    def interview_fun(self, cv_passed_seekers: list):
        current_hc = min(self.hc, len(cv_passed_seekers))


    def reply(self, x: Optional[dict] = None) -> dict:
        return Msg(self.name, None, role="assistant")
        # if self.memory:
        #     self.memory.add(x)

        # msg_hint = Msg("system", HINT_PROMPT, role="system")

        # prompt = self.model.format(
        #     self.memory.get_memory(),
        #     msg_hint,
        # )

        # response = self.model(
        #     prompt,
        #     parse_func=parse_func,
        #     max_retries=3,
        # ).raw

        # # For better presentation, we print the response proceeded by
        # # json.dumps, this msg won't be recorded in memory
        # self.speak(
        #     Msg(
        #         self.name,
        #         json.dumps(response, indent=4, ensure_ascii=False),
        #         role="assistant",
        #     ),
        # )

        # if self.memory:
        #     self.memory.add(Msg(self.name, response, role="assistant"))

        # # Hide thought from the response
        # return Msg(self.name, response["move"], role="assistant")