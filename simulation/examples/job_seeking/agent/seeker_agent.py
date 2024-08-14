import random
import os
import requests
import jinja2
from loguru import logger

from agentscope.message import Msg
from agentscope.rpc import async_func

from simulation.helpers.base_agent import BaseAgent
from simulation.helpers.utils import *
from simulation.helpers.constants import *
from simulation.examples.job_seeking.env import JobSeekingEnv


scene_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_loader = jinja2.FileSystemLoader(os.path.join(scene_path, "prompts"))
env = jinja2.Environment(loader=file_loader)
Template = env.get_template("seeker_prompts.j2").module


SeekerAgentStates = [
    "idle",
    "determining if seeking",
    "determining search job number",
    "determining search jobs",
    "determining jobs to apply",
    "applying jobs",
    "interviewing",
    "making final decision",
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
        return (
            f"Name: {self.name}\n"
            f"CV: {self.cv}\n"
            f"Current Working Condition: {self.working_condition}\n"
        )


class SeekerAgent(BaseAgent):
    """seeker agent."""

    def __init__(
        self,
        name: str,
        model_config_name: str,
        memory_config: dict,
        embedding_api: str,
        cv: str,
        trait: str,
        embedding: list,
        env: JobSeekingEnv,
        job_ids_pool: list[str] = [],
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            model_config_name=model_config_name,
        )
        self.memory_config = memory_config
        self.embedding_api = embedding_api
        self.memory = setup_memory(memory_config)
        self.memory.embedding_api = embedding_api
        self.memory.model = self.model
        self.job_ids_pool = job_ids_pool
        self.embedding = embedding
        self.env = env

        self.seeker = Seeker(name, cv, trait)
        self._update_profile()
        self._state = "idle"

    def _update_profile(self):
        self._profile = """
        Name: {name}
        CV: {cv}
        Trait: {trait}
        Working Condition: {working_condition}
        """.format(
            name=self.seeker.name,
            cv=self.seeker.cv,
            trait=self.seeker.trait,
            working_condition=self.seeker.working_condition,
        )

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_value):
        if hasattr(self, "backend_server_url"):
            if new_value not in SeekerAgentStates:
                raise ValueError(f"Invalid state: {new_value}")
            self._state = new_value
            url = f"{self.backend_server_url}/api/state"
            resp = requests.post(
                url, json={"agent_id": self.agent_id, "state": new_value}
            )
            if resp.status_code != 200:
                logger.error(f"Failed to set state: {self.agent_id} -- {new_value}")

    def _send_message(self, prompt, response):
        if hasattr(self, "backend_server_url"):
            url = f"{self.backend_server_url}/api/message"
            resp = requests.post(
                url,
                json={
                    "name": self.name,
                    "prompt": "\n".join([p["content"] for p in prompt]),
                    "completion": response.text,
                    "agent_type": type(self).__name__,
                    "agent_id": self.agent_id,
                },
            )
            if resp.status_code != 200:
                logger.error(f"Failed to send message: {self.agent_id}")

    @set_state("determining if seeking")
    def _determine_if_seeking(self, **kwargs):
        instruction = Template.determine_if_seeking_instruction()
        selection = ["yes", "no"]
        observation = Template.make_choice_observation(selection)
        msg = Msg("user", None, role="user")
        msg.instruction = instruction
        msg.observation = observation
        msg.selection_num = len(selection)
        response = selection[int(self.reply(msg)["content"])]
        return response

    @set_state("determining search job number")
    def _determine_search_job_number(self, **kwargs):
        """Set search job number."""
        instruction = Template.determine_search_job_number_instruction()
        Search_Job_Number = 5
        selection = list(range(1, Search_Job_Number + 1))
        observation = Template.make_choice_observation(selection)
        msg = Msg("user", None, role="user")
        msg.instruction = instruction
        msg.observation = observation
        msg.selection_num = len(selection)
        response = selection[int(self.reply(msg)["content"])]
        return response

    @set_state("determining search jobs")
    def _determine_search_jobs(self, search_job_number: int, **kwargs):
        search_job_indices = random.sample(
            range(len(self.job_ids_pool)), search_job_number
        )
        search_job_ids = [self.job_ids_pool[i] for i in search_job_indices]
        search_interviewer_agents = self.env.get_agents_by_ids(search_job_ids)
        interviewer_agent_infos = {
            agent.agent_id: agent for agent in search_interviewer_agents
        }
        for agent in interviewer_agent_infos.values():
            agent.job = agent.get_attr("job")

        return interviewer_agent_infos

    @set_state("determining jobs to apply")
    def _determine_apply_job(self, interviewer_agent_infos: dict, **kwargs):
        """Determine which jobs to apply."""
        instruction = Template.determine_apply_jobs_instruction()
        apply_interviewer_agent_infos = {}
        selection = ["yes", "no"]
        for job_id, agent in interviewer_agent_infos.items():
            job_info = agent.job
            observation = Template.determine_apply_jobs_observation(job_info, selection)
            msg = Msg("user", None, role="user")
            msg.instruction = instruction
            msg.observation = observation
            msg.selection_num = len(selection)
            response = selection[int(self.reply(msg)["content"])]
            if response == "yes":
                apply_interviewer_agent_infos[job_id] = agent

        return apply_interviewer_agent_infos

    @set_state("applying jobs")
    def _apply_job(self, apply_interviewer_agent_infos: dict, **kwargs):
        """Apply jobs."""
        results = []
        for agent in apply_interviewer_agent_infos.values():
            result = agent.screening_cv(str(self.seeker))

        cv_passed_interviewer_agent_infos = {}
        for (agent_id, agent), result in zip(
            apply_interviewer_agent_infos.items(), results
        ):
            if "yes" in result.get():
                cv_passed_interviewer_agent_infos[agent_id] = agent
        if len(cv_passed_interviewer_agent_infos) > 0:
            self.observe(
                get_assistant_msg(
                    Template.apply_job_observation(cv_passed_interviewer_agent_infos)
                )
            )

        return cv_passed_interviewer_agent_infos

    @set_state("interviewing")
    def _interview_fun(self, cv_passed_interviewer_agent_infos: dict, **kwargs):
        """Interview."""
        MAX_INTERVIEW_ROUND = 1
        offer_interviewer_agent_infos = {}
        for agent_id, agent in cv_passed_interviewer_agent_infos.items():
            format_profile = PROFILE_BEGIN + self._profile + PROFILE_END
            instruction = Template.interview_opening_instruction()
            format_instruction = INSTRUCTION_BEGIN + instruction + INSTRUCTION_END
            memory = self.memory.get_memory(get_assistant_msg(instruction))
            format_memory = (
                MEMORY_BEGIN + "\n- ".join([m["content"] for m in memory]) + MEMORY_END
            )
            observation = "Seeker:"
            answer = self.model(
                self.model.format(
                    Msg(
                        "user",
                        format_instruction
                        + format_profile
                        + format_memory
                        + observation,
                        role="user",
                    )
                )
            ).text

            for _ in range(MAX_INTERVIEW_ROUND + 1):
                observation += answer
                if _ == MAX_INTERVIEW_ROUND:
                    result = agent.interview(
                        msg=Msg(
                            "user",
                            observation,
                            role="user",
                            end=True,
                        )
                    )
                    break
                else:
                    observation += "\nInterviewer:"
                    question = agent.interview(
                        msg=Msg(
                            "user",
                            observation,
                            role="user",
                        )
                    )

                observation += question + "\nSeeker:"
                if _ == MAX_INTERVIEW_ROUND - 1:
                    msg = get_assistant_msg()
                    msg.instruction = instruction
                    msg.observation = observation
                    answer = self.reply(msg)["content"]
                else:
                    memory = self.memory.get_memory(
                        get_assistant_msg(instruction + observation)
                    )
                    format_memory = (
                        MEMORY_BEGIN
                        + "\n- ".join([m["content"] for m in memory])
                        + MEMORY_END
                    )
                    answer = self.model(
                        self.model.format(
                            Msg(
                                "user",
                                format_instruction
                                + format_profile
                                + format_memory
                                + observation,
                                role="user",
                            )
                        )
                    ).text

            if "yes" in result:
                offer_interviewer_agent_infos[agent_id] = agent
                self.observe(
                    get_assistant_msg(
                        Template.interview_observation(agent.job, True)
                    )
                )
            else:
                self.observe(
                    get_assistant_msg(
                        Template.interview_observation(agent.job, False)
                    )
                )

        return offer_interviewer_agent_infos

    @set_state("making final decision")
    def _make_final_decision(self, offer_interviewer_agent_infos: dict, **kwargs):
        """Make decision."""
        if len(offer_interviewer_agent_infos) == 0:
            return -1

        instruction = Template.make_final_decision_instruction()
        selection = ["-1"] + list(offer_interviewer_agent_infos.keys())
        observation = Template.make_choice_observation(selection)
        msg = Msg("user", None, role="user")
        msg.instruction = instruction
        msg.observation = observation
        msg.selection_num = len(selection)
        response = selection[int(self.reply(msg)["content"])]

        if response != "-1":
            final_job = offer_interviewer_agent_infos[response].job
            self.seeker.working_condition = (
                "Position Name: " + final_job["Position Name"]
            )
            self._update_profile()

        results = []
        for agent_id, agent in offer_interviewer_agent_infos.items():
            results.append(agent.receive_notification(self.seeker.name, agent_id == response))

        for result in results:
            result.get()

        return response

    @async_func
    def run(self, **kwargs):
        if self.seeker.working_condition != "unemployed":
            if "no" in self._determine_if_seeking():
                return

        search_job_number = self._determine_search_job_number()
        logger.info(f"Search job number: {search_job_number}")

        interviewer_agent_infos = self._determine_search_jobs(search_job_number)
        logger.info(f"Search jobs: {interviewer_agent_infos.keys()}")

        apply_interviewer_agent_infos = self._determine_apply_job(
            interviewer_agent_infos
        )
        logger.info(f"Apply jobs: {apply_interviewer_agent_infos.keys()}")

        cv_passed_interviewer_agent_infos = self._apply_job(
            apply_interviewer_agent_infos
        )
        logger.info(f"CV passed jobs: {cv_passed_interviewer_agent_infos.keys()}")

        offer_interviewer_agent_infos = self._interview_fun(
            cv_passed_interviewer_agent_infos
        )
        logger.info(f"Offer jobs: {offer_interviewer_agent_infos.keys()}")

        final_job_id = self._make_final_decision(offer_interviewer_agent_infos)
        logger.info(f"Final job: {final_job_id}")

        return final_job_id
