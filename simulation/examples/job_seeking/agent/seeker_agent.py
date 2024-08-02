import random
import os
from typing import Optional
import jinja2
from typing import Union
from typing import Sequence

from agentscope.agents import AgentBase
from agentscope.agents.agent import DistConf
from agentscope.message import Msg
from agentscope.models import ModelResponse
from agentscope.models import load_model_by_config_name

from simulation.examples.job_seeking.utils.utils import extract_dict
from simulation.helpers.message import MessageUnit, StateUnit, message_manager
from simulation.helpers.utils import setup_memory
from simulation.examples.job_seeking.environment import Environment


scene_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_loader = jinja2.FileSystemLoader(os.path.join(scene_path, "prompts"))
env = jinja2.Environment(loader=file_loader)
Template = env.get_template("seeker_prompts.j2").module


SeekerAgentStates = [
    "idle", 
    "determining status",
    "determining search job number",
    "determining search jobs",
    "determining jobs to apply",
    "applying jobs",
    "interviewing",
    "making final decision",
    "external interviewing",
]

def set_state(flag: str):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            init_state = self.state
            self.state = flag
            try:
                return func(self, *args, **kwargs)
            finally:
                self.state = init_state
        return wrapper
    return decorator


class Seeker(object):
    def __init__(self, name: str, cv: str, trait: str):
        self.name = name
        self.cv = cv
        self.trait = trait
        self.working_condition = "unemployed"

    def __str__(self):
        return f"Name: {self.name}\n" \
            f"CV: {self.cv}\n" \
            f"Current Working Condition: {self.working_condition}\n"


class SeekerAgent(AgentBase):
    """seeker agent."""

    name: str  # Name of the seeker
    model_config_name: str  # Model config name
    seeker: Seeker  # Seeker object
    system_prompt: Msg  # System prompt
    job_ids_pool: list  # Job ids pool

    def __init__(
        self,
        name: str,
        model_config_name: str,
        cv: str,
        trait: str,
        env: Environment,
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
        self.env = env
        self.seeker = Seeker(name, cv, trait)
        self.system_prompt = Msg(
            "system", Template.system_prompt(self.seeker), role="system"
        )
        
        self._state = "idle"

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        state.pop("model")
        memory_state = self.memory.__dict__.copy()
        try:
            memory_state["model"] = None
            memory_state["embedding_model"] = None
        except:
            pass
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
        if new_value not in SeekerAgentStates:
            raise ValueError(f"Invalid state: {new_value}")
        self._state = new_value
        message_manager.add_state(StateUnit(agent_id=self.id, state=new_value))

    def set_id(self, id: int):
        self.id = id
        self.seeker.id = id

    def get_id(self):
        return self.id

    def set_embedding(self, embedding_model):
        self.embedding = embedding_model.encode(
            str(self.seeker), normalize_embeddings=True
        )

    def send_message(self, prompt, response):
        message_manager.add_message(MessageUnit(
            name=self.name,
            prompt="\n".join([p["content"] for p in prompt]),
            completion=response.text,
            agent_type=type(self).__name__,
            agent_id=self.get_id(),
        ))

    @set_state("determining status")
    def determine_if_seeking_fun(self, **kwargs):
        msg = Msg("user", Template.determine_if_seeking_prompt(), role="user")
        response = self.reply(msg)
        return response.content

    @set_state("determining search job number")
    def determine_search_job_number_fun(self, **kwargs):
        """Set search job number."""
        msg = Msg("user", Template.determine_search_job_number_prompt(), role="user")
        response = self.reply(msg)
        search_job_number = int(extract_dict(response.content)["number"])
        return search_job_number
    
    @set_state("determining search jobs")
    def determine_search_jobs_fun(self, search_job_number: int, **kwargs):
        search_job_ids = random.sample(self.job_ids_pool, search_job_number)
        search_jobs = self.env.get_jobs(search_job_ids)

        self.observe(Msg(
            "assistant", Template.determine_search_jobs_memory(search_jobs), role="assistant"
        ))
        return search_jobs

    @set_state("determining jobs to apply")
    def determine_apply_job_fun(self, search_jobs: list, **kwargs):
        """Determine which jobs to apply."""
        msg = Msg("user", Template.determine_apply_jobs_prompt(search_jobs), role="user")
        response = self.reply(msg)
        apply_job_ids = list(map(int, extract_dict(response.content)["apply_jobs"]))
        apply_jobs = [job for job in search_jobs if job.id in apply_job_ids]

        return apply_jobs

    @set_state("applying jobs")
    def apply_job_fun(self, apply_jobs: list, **kwargs):
        """Apply jobs."""
        cv_passed_jobs = []
        for job in apply_jobs:
            response = self.env.apply_job(self.seeker, job)
            if "yes" in response.content:
                cv_passed_jobs.append(job)

        self.observe(Msg(
            "assistant", Template.apply_job_observation(cv_passed_jobs), role="assistant"
        ))

        return cv_passed_jobs

    @set_state("interviewing")
    def interview_fun(self, cv_passed_jobs: list, **kwargs):
        """Interview."""
        offer_jobs = []
        for job in cv_passed_jobs:
            response = self.env.interview(self.seeker, job)
            if "yes" in response["content"]:
                offer_jobs.append(job)
                self.observe(Msg(
                    "assistant", Template.interview_observation(job, True), role="assistant"
                ))
            else:
                self.observe(Msg(
                    "assistant", Template.interview_observation(job, False), role="assistant"
                ))

        return offer_jobs

    @set_state("making final decision")
    def make_final_decision_fun(self, offer_jobs: list, **kwargs):
        """Make decision."""
        if len(offer_jobs) > 0:
            msg = Msg("user", Template.make_final_decision_prompt(offer_jobs), role="user")
            response = self.reply(msg)
            final_job_id = int(extract_dict(response.content)["final_decision"])
        
        for job in offer_jobs:
            self.env.notify_interviewer(self.seeker, job, final_job_id == job.id)

    @set_state("external interviewing")
    def external_interview_fun(self, query, **kwargs):
        query_msg = Msg("assistant", query, role="assistant")
        memory_msg = self.memory.get_memory(query_msg)
        msg = Msg("assistant", "\n".join([p.content for p in memory_msg]) + query, "assistant")
        prompt = self.model.format(self.system_prompt, msg)
        response = self.model(prompt)
        return response.text
    
    def run_fun(self, **kwargs):
        if self.seeker.working_condition != "unemployed":
            if "no" in self.determine_if_seeking_fun(): return
        
        search_job_number = self.determine_search_job_number_fun()
        search_jobs = self.determine_search_jobs_fun(search_job_number)
        apply_jobs = self.determine_apply_job_fun(search_jobs)
        cv_passed_jobs = self.apply_job_fun(apply_jobs)
        offer_jobs = self.interview_fun(cv_passed_jobs)
        self.make_final_decision_fun(offer_jobs)

    def reply(self, x: Optional[Union[Msg, Sequence[Msg]]] = None) -> Msg:
        if x and x.get("fun", None):
            return getattr(self, f"{x.fun}_fun")(**getattr(x, "params", {}))
        else:
            memory = self.memory.get_memory(x)
            if x:
                self.memory.add(x)
                msg = Msg("user", "\n".join([p.content for p in memory]) + x.content, "user")
            else:
                msg = Msg("user", "\n".join([p.content for p in memory]), "user")
            prompt = self.model.format(self.system_prompt, msg)
            response = self.model(prompt)
            self.send_message(prompt, response)
            msg = Msg(self.name, response.text, role="user")
            self.memory.add(msg)
            return msg
        
