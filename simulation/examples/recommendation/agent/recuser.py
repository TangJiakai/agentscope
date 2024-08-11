import random
import os
import requests
import jinja2
from loguru import logger

from agentscope.message import Msg
from agentscope.rpc.rpc_agent_client import RpcAgentClient

from simulation.helpers.base_agent import BaseAgent
from simulation.examples.job_seeking.utils.utils import *
from simulation.helpers.utils import *
from simulation.helpers.constants import *


scene_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_loader = jinja2.FileSystemLoader(os.path.join(scene_path, "prompts"))
env = jinja2.Environment(loader=file_loader)
Template = env.get_template("seeker_prompts.j2").module


SeekerAgentStates = [
    "idle",
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


class User(object):
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


class RecUserAgent(BaseAgent):
    """recuser agent."""

    def __init__(
        self,
        name: str,
        model_config_name: str,
        memory_config: dict,
        embedding_api: str,
        cv: str,
        trait: str,
        embedding: list,
        env_agent: BaseAgent,
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
        self.env_agent = env_agent

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
    def _determine_if_seeking_fun(self, **kwargs):
        instruction = Template.determine_if_seeking_instruction()
        selection=["yes", "no"]
        observation = Template.make_choice_observation(selection)
        msg = Msg("user", None, role="user")
        msg.instruction = instruction
        msg.observation = observation
        msg.selection_num = len(selection)
        response = selection[int(self.reply(msg)["content"])]
        return response

    @set_state("determining search job number")
    def _determine_search_job_number_fun(self, **kwargs):
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
    def _determine_search_jobs_fun(self, search_job_number: int, **kwargs):
        search_job_indices = random.sample(
            range(len(self.job_ids_pool)), search_job_number
        )
        search_job_ids = [self.job_ids_pool[i] for i in search_job_indices]
        search_interviewer_agent_distribution_infos = self.env_agent(
            Msg(
                "assistant",
                None,
                role="assistant",
                fun="get_agent_distribution_infos",
                params={"agent_ids": search_job_ids},
            )
        )["content"]
        interviewer_agent_infos = {
            agent_id: {"agent_client": RpcAgentClient(**agent_dist_info)}
            for agent_id, agent_dist_info in search_interviewer_agent_distribution_infos.items()
        }
        for agent_info in interviewer_agent_infos.values():
            agent_info["job"] = rpc_client_post(
                agent_info["agent_client"], fun="get_attr", params={"attr": "job"}
            )

        for agent_info in interviewer_agent_infos.values():
            agent_info["job"] = rpc_client_get(
                agent_info["agent_client"], agent_info["job"]
            )["content"]

        return interviewer_agent_infos

    @set_state("determining jobs to apply")
    def _determine_apply_job_fun(self, interviewer_agent_infos: dict, **kwargs):
        """Determine which jobs to apply."""
        instruction = Template.determine_apply_jobs_instruction()
        observation = Template.determine_apply_jobs_observation(interviewer_agent_infos)
        msg = Msg("user", None, role="user")
        msg.instruction = instruction
        msg.observation = observation
        apply_ids = extract_agent_id(extract_dict(self.reply(msg)["content"])["result"])
        apply_ids = [apply_ids] if isinstance(apply_ids, str) else apply_ids

        if "-1" in apply_ids:
            return {}
        valid_ids = set(interviewer_agent_infos.keys())
        apply_interviewer_agent_infos = {
            agent_id: interviewer_agent_infos[agent_id]
            for agent_id in apply_ids
            if agent_id in valid_ids
        }

        return apply_interviewer_agent_infos

    @set_state("applying jobs")
    def _apply_job_fun(self, apply_interviewer_agent_infos: dict, **kwargs):
        """Apply jobs."""
        results = []
        for agent_info in apply_interviewer_agent_infos.values():
            results.append(
                rpc_client_post(
                    agent_info["agent_client"],
                    fun="screening_cv",
                    params={"seeker_info": str(self.seeker)},
                )
            )

        cv_passed_interviewer_agent_infos = {}
        for (agent_id, agent_info), result in zip(
            apply_interviewer_agent_infos.items(), results
        ):
            if "yes" in rpc_client_get(agent_info["agent_client"], result)["content"]:
                cv_passed_interviewer_agent_infos[agent_id] = agent_info
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
        for agent_id, agent_info in cv_passed_interviewer_agent_infos.items():
            format_profile = PROFILE_BEGIN + self._profile + PROFILE_END
            instruction = Template.interview_opening_instruction()
            format_instruction = INSTRUCTION_BEGIN + instruction + INSTRUCTION_END
            memory = self.memory.get_memory(get_assistant_msg(instruction))
            format_memory = MEMORY_BEGIN + "\n- ".join([m["content"] for m in memory]) + MEMORY_END
            observation = "Seeker:"
            format_observation = OBSERVATION_BEGIN + observation + OBSERVATION_END
            answer = self.model(self.model.format(Msg(
                "user",
                format_instruction + format_profile + format_memory + format_observation,
                role="user",
            )))
            
            for _ in range(MAX_INTERVIEW_ROUND + 1):
                observation += answer["content"]
                if _ == MAX_INTERVIEW_ROUND:
                    result = rpc_client_post_and_get(
                        agent_info["agent_client"],
                        fun="interview",
                        params={
                            "msg": Msg(
                                "user",
                                observation,
                                role="user",
                                end=True,
                            )
                        }
                    )
                    break
                else:
                    observation += "\nInterviewer:"
                    question = rpc_client_post_and_get(
                        agent_info["agent_client"],
                        fun="interview",
                        params={
                            "msg": Msg(
                                "user",
                                observation,
                                role="user",
                            )
                        },
                    )
                
                observation += question["content"] + "\nSeeker:"
                if _ == MAX_INTERVIEW_ROUND - 1:
                    msg = get_assistant_msg()
                    msg.instruction = instruction
                    msg.observation = observation
                    answer = self.reply(msg)
                else:
                    memory = self.memory.get_memory(get_assistant_msg(instruction + observation))
                    format_memory = MEMORY_BEGIN + "\n- ".join([m["content"] for m in memory]) + MEMORY_END
                    format_observation = OBSERVATION_BEGIN + observation + OBSERVATION_END
                    answer = self.model(self.model.format(Msg(
                        "user",
                        format_instruction + format_profile + format_memory + format_observation,
                        role="user",
                    )))

            if "yes" in result["content"]:
                offer_interviewer_agent_infos[agent_id] = agent_info
                self.observe(
                    get_assistant_msg(
                        Template.interview_observation(agent_info["job"], True)
                    )
                )
            else:
                self.observe(
                    get_assistant_msg(
                        Template.interview_observation(agent_info["job"], False)
                    )
                )

        return offer_interviewer_agent_infos

    @set_state("making final decision")
    def _make_final_decision_fun(self, offer_interviewer_agent_infos: dict, **kwargs):
        """Make decision."""
        if len(offer_interviewer_agent_infos) == 0:
            return -1
        
        instruction = Template.make_final_decision_instruction()
        selection = ['-1'] + list(offer_interviewer_agent_infos.keys())
        observation = Template.make_choice_observation(selection)
        msg = Msg("user", None, role="user")
        msg.instruction = instruction
        msg.observation = observation
        msg.selection_num = len(selection)
        response = selection[int(self.reply(msg)["content"])]

        if response != '-1':
            final_job = offer_interviewer_agent_infos[response]["job"]
            self.seeker.working_condition = "Position Name: " + final_job["Position Name"]
            self._update_profile()

        results = []
        for agent_id, agent_info in offer_interviewer_agent_infos.items():
            results.append(
                rpc_client_post(
                    agent_info["agent_client"],
                    fun="receive_notification",
                    params={
                        "seeker_name": self.seeker.name,
                        "is_accept": agent_id == response,
                    },
                )
            )

        for agent_info, res in zip(offer_interviewer_agent_infos.values(), results):
            rpc_client_get(agent_info["agent_client"], res)

        return response

    def run_fun(self, **kwargs):
        if self.seeker.working_condition != "unemployed":
            if "no" in self._determine_if_seeking_fun():
                return

        search_job_number = self._determine_search_job_number_fun()
        interviewer_agent_infos = self._determine_search_jobs_fun(search_job_number)
        apply_interviewer_agent_infos = self._determine_apply_job_fun(
            interviewer_agent_infos
        )
        cv_passed_interviewer_agent_infos = self._apply_job_fun(
            apply_interviewer_agent_infos
        )
        offer_interviewer_agent_infos = self._interview_fun(
            cv_passed_interviewer_agent_infos
        )
        final_job_id = self._make_final_decision_fun(offer_interviewer_agent_infos)

        return get_assistant_msg(final_job_id)