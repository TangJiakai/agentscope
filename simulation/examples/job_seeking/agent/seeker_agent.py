import random
import os
import requests
from typing import Optional
import jinja2
from typing import Union
from typing import Sequence
from loguru import logger

from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.models import load_model_by_config_name
from agentscope.rpc.rpc_agent_client import RpcAgentClient

from simulation.examples.job_seeking.utils.utils import *
from simulation.helpers.message import MessageUnit
from simulation.helpers.utils import *


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

backend_server_url = "http://localhost:39000"


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
    def __init__(self, name: str, cv: str, trait: str, working_condition="unemployed"):
        self.name = name
        self.cv = cv
        self.trait = trait
        self.working_condition = working_condition

    def __str__(self):
        return (
            f"Name: {self.name}\n"
            f"CV: {self.cv}\n"
            f"Current Working Condition: {self.working_condition}\n"
        )


class SeekerAgent(AgentBase):
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
        env_agent: AgentBase,
        job_ids_pool: list[str] = [],
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            model_config_name=model_config_name,
        )
        self.model_config_name = model_config_name
        self.memory_config = memory_config
        self.embedding_api = embedding_api
        self.memory = setup_memory(memory_config)
        self.memory.embedding_api = embedding_api
        self.memory.model = self.model
        self.job_ids_pool = job_ids_pool
        self.embedding = embedding
        self.env_agent = env_agent

        self.seeker = Seeker(name, cv, trait)

        self._update_system_prompt()
        self._state = "idle"

    def _update_system_prompt(self):
        self.sys_prompt = Msg(
            "system", Template.system_prompt(self.seeker), role="system"
        )

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        state.pop("model")
        memory_state = self.memory.__dict__.copy()
        memory_state["model"] = None
        state["memory"] = memory_state
        return state

    def __setstate__(self, state: object) -> None:
        self.__dict__.update(state)
        self.memory = setup_memory(self.memory_config)
        self.memory.__dict__.update(state["memory"])
        self.model = load_model_by_config_name(self.model_config_name)
        self.memory.model = self.model

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_value):
        if new_value not in SeekerAgentStates:
            raise ValueError(f"Invalid state: {new_value}")
        self._state = new_value
        url = f"{backend_server_url}/api/state"
        resp = requests.post(url, json={"agent_id": self.agent_id, "state": new_value})
        if resp.status_code != 200:
            logger.error(f"Failed to set state: {self.agent_id} -- {new_value}")

    def _send_message(self, prompt, response):
        url = f"{backend_server_url}/api/message"
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

    def set_attr_fun(self, attr: str, value, **kwargs):
        setattr(self, attr, value)
        return get_assistant_msg("success")

    def get_attr_fun(self, attr):
        if attr == "sys_prompt":
            return self.sys_prompt
        return get_assistant_msg(getattr(self, attr))

    @set_state("determining status")
    def _determine_if_seeking_fun(self, **kwargs):
        msg = Msg("user", Template.determine_if_seeking_prompt(), role="user")
        response = self.reply(msg)
        return response.content

    @set_state("determining search job number")
    def _determine_search_job_number_fun(self, **kwargs):
        """Set search job number."""
        msg = Msg("user", Template.determine_search_job_number_prompt(), role="user")
        response = self.reply(msg)["content"]
        search_job_number = int(extract_dict(response)["result"])
        logger.info("Search job number: {}", search_job_number)
        return search_job_number

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

        self.observe(
            get_assistant_msg(
                Template.determine_search_jobs_memory(interviewer_agent_infos)
            )
        )

        return interviewer_agent_infos

    @set_state("determining jobs to apply")
    def _determine_apply_job_fun(self, interviewer_agent_infos: dict, **kwargs):
        """Determine which jobs to apply."""
        msg = Msg(
            "user",
            Template.determine_apply_jobs_prompt(interviewer_agent_infos),
            role="user",
        )
        apply_ids = extract_agent_id(extract_dict(self.reply(msg)["content"])["result"])
        apply_ids = [apply_ids] if isinstance(apply_ids, str) else apply_ids
        logger.info("Apply jobs: {}", apply_ids)
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
            moderator_opening_msg = get_assistant_msg(
                Template.interview_opening_statement(self.seeker, agent_info["job"])
            )
            answer = self.reply(moderator_opening_msg)
            question = rpc_client_post_and_get(
                agent_info["agent_client"],
                fun="start_interview",
                params={
                    "msg": get_assistant_msg(
                        content=moderator_opening_msg["content"]
                        + answer["content"]
                        + Template.interview_turn_taking("Interviewer")
                    )
                },
            )
            answer = self.reply(
                get_assistant_msg(
                    question["content"] + Template.interview_turn_taking("Seeker")
                )
            )
            for _ in range(MAX_INTERVIEW_ROUND - 1):
                question = rpc_client_post_and_get(
                    agent_info["agent_client"],
                    msg=get_assistant_msg(
                        content=answer["content"]
                        + Template.interview_turn_taking("Interviewer")
                    ),
                )
                answer = self.reply(
                    Msg(
                        "user",
                        question["content"] + Template.interview_turn_taking("Seeker"),
                        role="user",
                    )
                )
            result = rpc_client_post_and_get(
                agent_info["agent_client"],
                fun="end_interview",
                params={"msg": get_assistant_msg(content=answer["content"])},
            )
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
        final_job_id = "-1"
        if len(offer_interviewer_agent_infos) > 0:
            msg = Msg(
                "user",
                Template.make_final_decision_prompt(offer_interviewer_agent_infos),
                role="user",
            )
            final_job_id = extract_agent_id(
                extract_dict(self.reply(msg)["content"])["result"]
            )

        if final_job_id == "-1":
            self.seeker.working_condition = offer_interviewer_agent_infos[final_job_id][
                "job"
            ]["Position Name"]
            self._update_system_prompt()

        results = []
        for agent_id, agent_info in offer_interviewer_agent_infos.items():
            results.append(
                rpc_client_post(
                    agent_info["agent_client"],
                    fun="receive_notification",
                    params={
                        "seeker_name": self.seeker.name,
                        "is_accept": agent_id == final_job_id,
                    },
                )
            )

        for agent_info, res in zip(offer_interviewer_agent_infos.values(), results):
            rpc_client_get(agent_info["agent_client"], res)

        return final_job_id

    @set_state("external interviewing")
    def external_interview_fun(self, query, **kwargs):
        query_msg = get_assistant_msg(query)
        memory_msg = self.memory.get_memory(query_msg)
        msg = get_assistant_msg("\n".join([p["content"] for p in memory_msg]) + query)
        prompt = self.model.format(self.sys_prompt, msg)
        response = self.model(prompt)
        return Msg(self.name, response.text, "user")

    def get_memory_fun(self, **kwargs):
        return get_assistant_msg(self.memory.get_memory())

    def post_intervention_fun(self, intervention: str, **kwargs):
        self.memory.add(Msg("assistant", intervention, role="assistant"))
        return get_assistant_msg("success")

    def run_fun(self, **kwargs):
        if self.seeker.working_condition != "unemployed":
            if "no" in self.determine_if_seeking_fun():
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

    def reply(self, x: Optional[Union[Msg, Sequence[Msg]]] = None) -> Msg:
        if x and x.get("fun", None):
            return getattr(self, f"{x.fun}_fun")(**getattr(x, "params", {}))
        else:
            memory = self.memory.get_memory(x)
            if x:
                self.memory.add(x)
                msg = Msg(
                    "user",
                    "\n".join([p["content"] for p in memory]) + x["content"],
                    "user",
                )
            else:
                msg = Msg("user", "\n".join([p["content"] for p in memory]), "user")
            prompt = self.model.format(self.sys_prompt, msg)
            response = self.model(prompt)
            self._send_message(prompt, response)
            msg = Msg(self.name, response.text, role="user")
            self.memory.add(msg)
            return msg
