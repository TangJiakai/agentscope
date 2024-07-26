import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import dill
from loguru import logger
import numpy as np
from copy import deepcopy
import random
from sentence_transformers import SentenceTransformer
import faiss

import agentscope
from agentscope.file_manager import file_manager
from agentscope.message import Msg

from simulation.helpers.simulator import BaseSimulator
from simulation.helpers.utils import load_yaml, load_json, save_configs
from simulation.helpers.events import check_pause, play_event, stop_event
from simulation.helpers.message import message_manager
from simulation.helpers.constants import *
from agentscope.constants import _DEFAULT_DIR

from simulation.examples.job_seeking.agent import SeekerAgent, JobAgent, CompanyAgent
from simulation.helpers.utils import setup_memory

CUR_ROUND = 1
SEEKER_AGENT_CONFIG = "seeker_agent_configs.json"
JOB_AGENT_CONFIG = "job_agent_configs.json"
COMPANY_AGENT_CONFIG = "company_agent_configs.json"

scene_path = os.path.dirname(os.path.abspath(__file__))


class Simulator(BaseSimulator):
    def __init__(self):
        super().__init__()
        self.config = load_yaml(os.path.join(scene_path, CONFIG_DIR, SIMULATION_CONFIG))
        
        global CUR_ROUND
        self.cur_round = CUR_ROUND
        from agentscope import constants
        constants._DEFAULT_DIR = file_manager.dir = self.config["save_dir"]

        self._from_scratch()

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        state["embedding_model"] = None
        return state
    
    def __setstate__(self, state: object) -> None:
        self.__dict__.update(state)
        self._init_embedding_model()
        self._set_agent_models()

    def _from_scratch(self):
        self._init_agentscope()

        if self.config["load_simulator_path"] is not None:
            loaded_simulator = Simulator.load(self.config["load_simulator_path"])
            self.__dict__.update(loaded_simulator.__dict__)
            global CUR_ROUND
            CUR_ROUND = self.cur_round
        else:
            self._init_embedding_model()
            self._init_agents()
        
        save_configs(self.config)

    def _init_agentscope(self):
        agentscope.init(
            project=self.config["project_name"],
            save_code=False,
            save_api_invoke=False,
            model_configs=os.path.join(scene_path, CONFIG_DIR, self.config["model_configs_path"]),
            use_monitor=False,
            save_dir=self.config["save_dir"] if self.config["save_dir"] else _DEFAULT_DIR,
        )

    def _init_embedding_model(self):
        self.embedding_model = SentenceTransformer(
            os.path.join(scene_path, "../../", self.config["embedding_model_path"]), 
            device=f"cuda:{self.config['gpu']}" if self.config["gpu"] else "cpu",
        )

    def _set_agent_models(self):
        for agent in self.agents:
            if hasattr(agent.memory, "model"):
                agent.memory.model = agent.model
            if hasattr(agent.memory, "embedding_model"):
                agent.memory.embedding_model = self.embedding_model

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            return dill.load(f)

    def _init_agents(self):
        # Load configs
        seeker_configs = load_json(os.path.join(scene_path, CONFIG_DIR, SEEKER_AGENT_CONFIG))
        job_configs = load_json(os.path.join(scene_path, CONFIG_DIR, JOB_AGENT_CONFIG))
        company_configs = load_json(os.path.join(scene_path, CONFIG_DIR, COMPANY_AGENT_CONFIG))

        # Init agents
        self.seeker_agents = [
            SeekerAgent(
                **config["args"],
                distributed=self.config["distributed"], 
                **({"embedding_model_path": self.config["embedding_model_path"]} if self.config["embedding_model_path"] else {})
            )
            for config in seeker_configs
        ]
        self.job_agents = [
            JobAgent(**config["args"], 
                    distributed=self.config["distributed"],
                    **({"embedding_model_path": self.config["embedding_model_path"]} if self.config["embedding_model_path"] else {})
            )
            for config in job_configs
        ]
        self.company_agents = [
            CompanyAgent(
                **config["args"], 
                distributed=self.config["distributed"],
            )
            for config in company_configs
        ]
        self.agents = self.seeker_agents + self.job_agents + self.company_agents
        for id, agent in enumerate(self.agents):
            agent.set_id(id)

        # Init system prompt for all job agents
        name2company_agent = {agent.name: agent for agent in self.company_agents}
        for job_agent in self.job_agents:
            job_agent.init_system_prompt(name2company_agent[job_agent.job.company_name])

        for agent in self.seeker_agents + self.job_agents:
            memory_config = load_json(os.path.join(scene_path, CONFIG_DIR, MEMORY_CONFIG))
            agent.memory = setup_memory(memory_config)
            agent.memory_config = memory_config
            agent.set_embedding(self.embedding_model)

        # assign job_ids_pool for all seekers
        index = faiss.IndexFlatL2(self.embedding_model.get_sentence_embedding_dimension())
        index.add(np.array([agent.embedding for agent in self.job_agents]))
        for seeker_agent in self.seeker_agents:
            _, job_ids = index.search(np.array([seeker_agent.embedding]), self.config["pool_size"])
            seeker_agent.job_ids_pool = list(job_ids[0] + len(self.seeker_agents))

        self._set_agent_models()

    def _make_decision_round(self, seeker_agents):
        job_agents = self.job_agents

        for seeker_agent in seeker_agents:
            seeker_agent(
                Msg(
                    "assistant",
                    None,
                    fun="make_decision",
                    params={
                        "agents": self.agents,
                    },
                )
            )

        for seeker_agent in seeker_agents:
            if (
                seeker_agent.decision == 0
            ):  # No any offers, and continue to search for jobs
                seeker_agent.memory_info["final_decision"] = 3
                logger.info(
                    f"{seeker_agent.name} has no any offers, and continues to search for jobs."
                )
            elif seeker_agent.decision == 1:  # Accept the offer
                seeker_agent.memory_info["final_decision"] = (
                    1 if seeker_agent.memory_info["waiting_time"] == 0 else 2
                )
                seeker_agent.memory_info["final_offer"] = self.agents[
                    seeker_agent.final_offer_id
                ].job
                company_name = self.agents[seeker_agent.final_offer_id].job.company_name
                seeker_agent.update_job_condition(
                    company_name, self.agents[seeker_agent.final_offer_id].job
                )
                logger.info(
                    f"{seeker_agent.name} accepts the offer {self.agents[seeker_agent.final_offer_id].name}."
                )
            elif seeker_agent.decision == 2:  # Wait for the waitlist offer
                seeker_agent.memory_info["waiting_time"] += 1
                logger.info(
                    f"{seeker_agent.name} rejects all offers, and waits for {[self.agents[x].name for x in seeker_agent.wl_jobs_dict]}."
                )
            elif (
                seeker_agent.decision == 3
            ):  # Reject all offers and waiting list, and continue to search for jobs
                seeker_agent.memory_info["final_decision"] = (
                    4 if seeker_agent.memory_info["waiting_time"] == 0 else 5
                )
                logger.info(
                    f"{seeker_agent.name} rejects all offers and waiting list, and continues to search for jobs."
                )

        for seeker_agent in seeker_agents:
            seeker_id = seeker_agent.get_id()
            if seeker_agent.decision == 1:  # Accept the offer
                job_agent = self.agents[seeker_agent.final_offer_id]
                job_agent.job.hc -= 1
                job_agent.offer_seeker_ids.remove(seeker_id)
                job_agent.memory_info["final_offer_seeker"].append(seeker_agent.seeker)
            for job_id in seeker_agent.reject_offer_job_ids:
                job_agent = self.agents[job_id]
                job_agent.offer_seeker_ids.remove(seeker_id)
                if len(job_agent.wl_seeker_ids) > 0:
                    wl_seeker_id = job_agent.wl_seeker_ids.pop(0)
                    job_agent.offer_seeker_ids.append(wl_seeker_id)
            for job_id in seeker_agent.reject_wl_job_ids:
                job_agent = self.agents[job_id]
                job_agent.wl_seeker_ids.remove(seeker_id)
                if len(job_agent.wl_seeker_ids) > 0:
                    wl_seeker_id = job_agent.wl_seeker_ids.pop(0)
                    job_agent.offer_seeker_ids.append(wl_seeker_id)

            seeker_agent.offer_job_ids, seeker_agent.wl_jobs_dict = list(), dict()

        for job_agent in job_agents:
            job_id = job_agent.get_id()
            for seeker_id in job_agent.offer_seeker_ids:
                seeker_agent = self.agents[seeker_id]
                seeker_agent.offer_job_ids.append(job_id)
            wl_n = len(job_agent.wl_seeker_ids)
            for i, seeker_id in enumerate(job_agent.wl_seeker_ids):
                seeker_agent = self.agents[seeker_id]
                seeker_agent.wl_jobs_dict[job_id] = {"rank": i + 1, "wl_n": wl_n}

        for job_agent in job_agents:
            logger.info(
                f"{job_agent.name} offers {[self.agents[x].name for x in job_agent.offer_seeker_ids]}, waitlists {[self.agents[x].name for x in job_agent.wl_seeker_ids]}."
            )

    def _one_round(self):
        job_agents = self.job_agents
        seeker_agents = []
        # determine status for all seekers
        for seeker_agent in self.seeker_agents:
            if seeker_agent.seeker.status == "employed":
                seeker_agent(Msg("assistant", None, fun="determine_status"))
        for seeker_agent in self.seeker_agents:
            if seeker_agent.seeker.status == "unemployed" or seeker_agent.seeking:
                seeker_agents.append(seeker_agent)
            else:
                seeker_agent(
                    Msg(
                        "assistant",
                        None,
                        fun="add_memory",
                    )
                )
        check_pause()

        # determine search job number for all seekers
        for seeker_agent in seeker_agents:
            seeker_agent(Msg("assistant", None, fun="search_job_number"))
            seeker_agent.memory_info["search_job_number"] = (
                seeker_agent.search_job_number
            )
            logger.info(
                f"{seeker_agent.name} wants to search {seeker_agent.search_job_number} jobs."
            )
        check_pause()

        # search job ids for all seekers
        for seeker_agent in seeker_agents:
            seeker_agent.search_job_ids = random.sample(
                seeker_agent.job_ids_pool, seeker_agent.search_job_number
            )
            seeker_agent.memory_info["search_jobs"] = [
                self.agents[x].job for x in seeker_agent.search_job_ids
            ]
            logger.info(
                f"{seeker_agent.name} searches {[self.agents[x].name for x in seeker_agent.search_job_ids]} jobs."
            )

        # apply job for all seekers
        logger.info("[Seeker] Apply for jobs.")
        for seeker_agent in seeker_agents:
            seeker_agent(
                Msg(
                    "assistant",
                    None,
                    fun="apply_job",
                    params={
                        "search_jobs": [
                            self.agents[job_id].job
                            for job_id in seeker_agent.search_job_ids
                        ]
                    },
                )
            )
            seeker_agent.memory_info["apply_job_ids"] = seeker_agent.apply_job_ids

        for seeker_agent in seeker_agents:
            seeker_id = seeker_agent.get_id()
            for job_id in seeker_agent.apply_job_ids:
                job_agent = self.agents[job_id]
                job_agent.apply_seeker_ids.append(seeker_id)
        check_pause()

        # cv screening for all job agents
        logger.info("[Job] Screen cv from job seekers.")
        for job_agent in job_agents:
            job_agent(
                Msg(
                    "assistant",
                    None,
                    fun="cv_screening",
                    params={
                        "apply_seekers": [
                            self.agents[seeker_id].seeker
                            for seeker_id in job_agent.apply_seeker_ids
                        ],
                        "excess_cv_passed_n": self.config["excess_cv_passed_n"],
                    },
                )
            )
            job_agent.memory_info["apply_seekers"] = [
                self.agents[x].seeker for x in job_agent.apply_seeker_ids
            ]
            job_agent.memory_info["cv_passed_seeker_ids"] = (
                job_agent.cv_passed_seeker_ids
            )
            logger.info(
                f"{job_agent.name} passes the cv screening for {[self.agents[x].name for x in job_agent.cv_passed_seeker_ids]} seekers."
            )

        # update cv passed seeker ids for all job agents
        logger.info("[Seeker] Notify the result of cv screening.")
        for job_agent in job_agents:
            job_id = job_agent.get_id()
            for seeker_id in job_agent.cv_passed_seeker_ids:
                self.agents[seeker_id].cv_passed_job_ids.append(job_id)

        for seeker_agent in seeker_agents:
            seeker_agent.memory_info["cv_passed_job_ids"] = (
                seeker_agent.cv_passed_job_ids
            )
            logger.info(
                f"{seeker_agent.name} passes the cv screening for {[self.agents[x].name for x in seeker_agent.cv_passed_job_ids]} jobs."
            )
        check_pause()

        # make decision for all job agents
        logger.info("[Job] Decision the interview result.")
        for job_agent in job_agents:
            job_agent(
                Msg(
                    "assistant",
                    None,
                    fun="make_decision",
                    params={
                        "interview_seekers": [
                            self.agents[seeker_id].seeker
                            for seeker_id in job_agent.cv_passed_seeker_ids
                        ],
                        "wl_n": self.config["wl_n"],
                    },
                )
            )
            job_agent.memory_info["offer_seeker_ids"] = deepcopy(
                job_agent.offer_seeker_ids
            )
            job_agent.memory_info["wl_seeker_ids"] = deepcopy(job_agent.wl_seeker_ids)
            logger.info(
                f"{job_agent.name} offers {[self.agents[x].name for x in job_agent.offer_seeker_ids]}, waitlists {[self.agents[x].name for x in job_agent.wl_seeker_ids]}, and rejects {[self.agents[x].name for x in job_agent.reject_seeker_ids]}."
            )

        # update offer, wl, reject for all seeker agents
        for job_agent in job_agents:
            job_id = job_agent.get_id()
            for seeker_id in job_agent.offer_seeker_ids:
                self.agents[seeker_id].offer_job_ids.append(job_id)
            for i, seeker_id in enumerate(job_agent.wl_seeker_ids):
                self.agents[seeker_id].wl_job_dict[job_id] = {
                    "rank": i + 1,
                    "wl_n": len(job_agent.wl_seeker_ids),
                }
            for seeker_id in job_agent.reject_seeker_ids:
                self.agents[seeker_id].fail_job_ids.append(job_id)

        for seeker_agent in seeker_agents:
            seeker_agent.memory_info["initial_offer_job_ids"] = deepcopy(
                seeker_agent.offer_job_ids
            )
            seeker_agent.memory_info["initial_wl_jobs_dict"] = deepcopy(
                seeker_agent.wl_jobs_dict
            )
            logger.info(
                f"{seeker_agent.name} receives {len(seeker_agent.offer_job_ids)} offers, {len(seeker_agent.wl_jobs_dict)} waiting list, and {len(seeker_agent.fail_job_ids)} failed jobs."
            )
        check_pause()

        cur_seeker_agents = seeker_agents
        for r in range(1, self.config["make_decision_round_n"] + 1):
            logger.info(f"Make decision round {r} started")
            self._make_decision_round(cur_seeker_agents)
            cur_seeker_agents = [x for x in cur_seeker_agents if x.decision == 2]

            stop_flag = True
            for seeker_agent in cur_seeker_agents:
                if len(seeker_agent.offer_job_ids) > 0:
                    stop_flag = False
                    break
            if stop_flag:
                break
            check_pause()

        logger.info("[Seeker + Job] Add memory & Refresh information.")
        # update memory for all agents
        for agent in seeker_agents + job_agents:
            agent(Msg("assistant", None, fun="add_memory"))
            agent(Msg("assistant", None, fun="update"))
        check_pause()

    def run(self):
        play_event.set()
        
        message_manager.message_queue.put("Start simulation.")
        for r in range(self.cur_round, self.config["round_n"] + 1):
            logger.info(f"Round {r} started")
            self._one_round()
            self.save()
            if stop_event.is_set():
                message_manager.message_queue.put(f"Stop simulation by user at round {r}.")
                logger.info(f"Stop simulation by user at round {r}.")
                break
        message_manager.message_queue.put("Simulation finished.")
        logger.info("Simulation finished")

    def save(self):
        global CUR_ROUND
        save_path = os.path.join(file_manager.dir_root, f"ROUND-{self.cur_round}.pkl")
        self.cur_round = CUR_ROUND + 1
        CUR_ROUND = self.cur_round
        with open(save_path, "wb") as f:
            dill.dump(self, f)
        logger.info(f"Saved simulator to {save_path}")


if __name__ == "__main__":
    simulator = Simulator()
    simulator.run()
