from typing import Optional
from jinja2 import Environment, FileSystemLoader

from agentscope.agents import AgentBase
from agentscope.message import Msg

file_loader = FileSystemLoader("prompts")
env = Environment(loader=file_loader)
Template = env.get_template('interviewer_prompts.j2').module


class InterviewerAgent(AgentBase):
    """interviewer agent."""

    def __init__(
        self,
        name: str,
        model_config_name: str,
        job: str,
        jd: str,
        hc: int,
        sys_prompt: Optional[str] = None,
    ) -> None:
        sys_prompt = sys_prompt or Template.system_prompt
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
        )
        
        self.job = job
        self.jd = jd
        self.hc = hc

        self.memory.add(Msg("system", sys_prompt, role="system"))

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