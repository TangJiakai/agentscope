from typing import Optional
from jinja2 import Environment, FileSystemLoader
import json

from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.models import ModelResponse

file_loader = FileSystemLoader("prompts")
env = Environment(loader=file_loader)
Template = env.get_template('seeker_prompts.j2').module


class SeekerAgent(AgentBase):
    """seeker agent."""

    def __init__(
        self,
        name: str,
        model_config_name: str,
        cv: str,
        trait: str,
        status: str,
    ) -> None:
        super().__init__(
            name=name,
            model_config_name=model_config_name,
        )
        self.cv = cv
        self.trait = trait
        self.status = status
        self.system_prompt = Msg("system", Template.system_prompt(cv=cv, trait=trait, status=status), role="system")
    
    def search_job_number_fun(self):
        msg = Msg("user", Template.search_job_number_prompt(), role="user")
        prompt = self.model.format(self.system_prompt, self.memory.get_memory(self.recent_n), msg)

        def parse_func(response: ModelResponse) -> ModelResponse:
            res_dict = json.loads(response.text)
            if "number" in res_dict:
                return ModelResponse(raw=int(res_dict["number"]))
            else:
                raise ValueError(
                    f"Invalid response format in parse_func "
                    f"with response: {response.text}",
                )
        # print(prompt)
        response = self.model(prompt, parse_func=parse_func).raw
        # print(response)
        self.search_job_number = min(response, len(self.job_pool))

    def apply_job_fun(self, search_jobs: list):
        msg = Msg("user", Template.apply_jobs_prompt(self.search_job_number, search_jobs), role="user")
        prompt = self.model.format(self.system_prompt, self.memory.get_memory(self.recent_n), msg)

        def parse_func(response: ModelResponse) -> ModelResponse:
            res_dict = json.loads(response.text)
            if "apply_jobs" in res_dict:
                return ModelResponse(raw=list(map(int, res_dict["apply_jobs"])))
            else:
                raise ValueError(
                    f"Invalid response format in parse_func "
                    f"with response: {response.text}",
                )
        
        # print(prompt)
        response = self.model(prompt, parse_func=parse_func).raw
        # print(response)
        self.apply_jobs = response

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