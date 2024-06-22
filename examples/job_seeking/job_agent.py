from typing import Optional
from jinja2 import Environment, FileSystemLoader
import json

from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.models.response import ModelResponse

file_loader = FileSystemLoader("prompts")
env = Environment(loader=file_loader)
Template = env.get_template('job_prompts.j2').module


class JobAgent(AgentBase):
    """job agent."""

    def __init__(
        self,
        name: str,
        model_config_name: str,
        jd: str,
        hc: int,
    ) -> None:
        super().__init__(
            name=name,
            model_config_name=model_config_name
        )
        self.hc = hc
        self.system_prompt = Msg("system", Template.system_prompt(name, jd), role="system")

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

        response = self.model(prompt, parse_func=parse_func).raw
        self.cv_passed_seekers = response

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