import os
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


scene_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_loader = FileSystemLoader(os.path.join(scene_path, "prompts"))
env = Environment(loader=file_loader)
Template = env.get_template("job_prompts.j2").module


JobAgentStates = ["idle", "screening_cv", "making_decision"]


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


class JobAgent(AgentBase):
    """job agent."""

    name: str  # Name of the job
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
        self.memory_info = {
            "final_offer_seeker": [],
        }

        (
            self.cv_passed_seeker_ids,
            self.offer_seeker_ids,
            self.wl_seeker_ids,
            self.reject_seeker_ids,
        ) = (list(), list(), list(), list())
        self.apply_seeker_ids = list()
        self.update_variables = [
            self.cv_passed_seeker_ids,
            self.offer_seeker_ids,
            self.wl_seeker_ids,
            self.reject_seeker_ids,
            self.apply_seeker_ids,
        ]
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
        if new_value not in JobAgentStates:
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

    def init_system_prompt(self, company):
        self.system_prompt = Msg(
            "system", Template.system_prompt(self.job, company), role="system"
        )

    def cv_screening_fun(self, apply_seekers: list, excess_cv_passed_n: int):
        self.state = "screening_cv"
        cv_passed_hc = min(self.job.hc + excess_cv_passed_n, len(apply_seekers))
        if cv_passed_hc == 0:
            self.cv_passed_seeker_ids = list()
            return

        msg = Msg(
            "user",
            Template.screen_resumes_prompt(cv_passed_hc, apply_seekers),
            role="user",
        )
        prompt = self.model.format(self.system_prompt, self.memory.get_memory(msg), msg)

        def parse_func(response: ModelResponse) -> ModelResponse:
            message_manager.add_message(
                MessageUnit(
                    name=self.name,
                    query="\n".join([p["content"] for p in prompt]),
                    response=response.text,
                    agent_type=type(self).__name__,
                    agent_id=self.get_id(),
                )
            )
            try:
                res_dict = extract_dict(response.text)
                return ModelResponse(
                    raw=list(map(int, res_dict["cv_passed_seeker_ids"]))
                )
            except:
                raise ValueError(
                    f"Invalid response format in parse_func "
                    f"with response: {response.text}",
                )

        # print(prompt)
        response = self.model(prompt, parse_func=parse_func).raw
        # print(response)
        self.cv_passed_seeker_ids = response
        self.state = "idle"

    def make_decision_fun(self, interview_seekers: list, wl_n: int):
        self.state = "making_decision"
        offer_hc = min(self.job.hc, len(interview_seekers))
        wl_n = min(wl_n, len(interview_seekers) - offer_hc)
        if offer_hc == 0:
            self.offer_seeker_ids = list()
            self.wl_seeker_ids = list()
            self.reject_seeker_ids = [seeker.id for seeker in interview_seekers]
            return

        msg = Msg(
            "user",
            Template.make_decision(offer_hc, wl_n, interview_seekers),
            role="user",
        )
        prompt = self.model.format(self.system_prompt, self.memory.get_memory(msg), msg)

        def parse_func(response: ModelResponse) -> ModelResponse:
            message_manager.add_message(
                MessageUnit(
                    name=self.name,
                    query="\n".join([p["content"] for p in prompt]),
                    response=response.text,
                    agent_type=type(self).__name__,
                    agent_id=self.get_id(),
                )
            )
            try:
                res_dict = extract_dict(response.text)
                return ModelResponse(
                    raw={
                        "offer_seeker_ids": list(
                            map(int, res_dict["offer_seeker_ids"])
                        ),
                        "wl_seeker_ids": list(map(int, res_dict["wl_seeker_ids"])),
                    }
                )
            except:
                raise ValueError(
                    f"Invalid response format in parse_func "
                    f"with response: {response.text}",
                )

        # print(prompt)
        response = self.model(prompt, parse_func=parse_func).raw
        # print(response)
        self.offer_seeker_ids = response["offer_seeker_ids"]
        self.wl_seeker_ids = response["wl_seeker_ids"]
        self.reject_seeker_ids = list(
            set([seeker.id for seeker in interview_seekers])
            - set(self.offer_seeker_ids)
            - set(self.wl_seeker_ids)
        )
        self.state = "idle"

    def add_memory_fun(self):
        mem = Msg("assistant", Template.job_memory(self.memory_info), role="assistant")
        self.memory.add(mem)

    def update_fun(self):
        self.memory_info = {
            "final_offer_seeker": [],
        }
        for var in self.update_variables:
            var.clear()

    def interview(self, query):
        msg = Msg("user", query, role="user")
        tht = self.reflect(current_action=query)
        prompt = self.model.format(self.system_prompt, tht, msg)
        return self.model(prompt).text

    def reply(self, x: Optional[Union[Msg, Sequence[Msg]]] = None) -> Msg:
        fun = getattr(self, f"{x.fun}_fun")
        if hasattr(x, "params"):
            fun(**x.params)
        else:
            fun()
        return Msg(self.name, None, role="assistant")
