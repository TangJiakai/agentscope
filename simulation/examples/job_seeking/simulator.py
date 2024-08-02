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

from simulation.examples.job_seeking2.agent import SeekerAgent, InterviewerAgent, CompanyAgent
from simulation.helpers.utils import setup_memory
from simulation.examples.job_seeking2.environment import Environment

CUR_ROUND = 1
SEEKER_AGENT_CONFIG = "seeker_agent_configs.json"
INTERVIEWER_AGENT_CONFIG = "interviewer_agent_configs.json"
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
            self.env = Environment()
            self._init_agents()

        save_configs(self.config)

    def _init_agentscope(self):
        agentscope.init(
            project=self.config["project_name"],
            save_code=False,
            save_api_invoke=False,
            model_configs=os.path.join(
                scene_path, CONFIG_DIR, self.config["model_configs_path"]
            ),
            use_monitor=False,
            save_dir=self.config["save_dir"] if self.config["save_dir"] else _DEFAULT_DIR,
        )

    def _init_embedding_model(self):
        self.embedding_model = SentenceTransformer(
            os.path.join(scene_path, "../../", self.config["embedding_model_path"]),
            device=f"cuda:{self.config['gpu']}" if self.config["gpu"] else "cpu",
        )

    def _set_agent_models(self):
        for agent in self.env.agents:
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
        seeker_configs = load_json(
            os.path.join(scene_path, CONFIG_DIR, SEEKER_AGENT_CONFIG)
        )
        interviewer_configs = load_json(os.path.join(scene_path, CONFIG_DIR, INTERVIEWER_AGENT_CONFIG))
        company_configs = load_json(
            os.path.join(scene_path, CONFIG_DIR, COMPANY_AGENT_CONFIG)
        )

        # Init agents
        seeker_agents = [
            SeekerAgent(
                **config["args"],
                distributed=self.config["distributed"],
                env=self.env,
                **(
                    {"embedding_model_path": self.config["embedding_model_path"]}
                    if self.config["embedding_model_path"]
                    else {}
                ),
            )
            for config in seeker_configs
        ]
        interviewer_agents = [
            InterviewerAgent(
                **config["args"],
                distributed=self.config["distributed"],
                env=self.env,
                **(
                    {"embedding_model_path": self.config["embedding_model_path"]}
                    if self.config["embedding_model_path"]
                    else {}
                ),
            )
            for config in interviewer_configs
        ]
        company_agents = [
            CompanyAgent(
                **config["args"],
                distributed=self.config["distributed"],
                env=self.env,
                **(
                    {"embedding_model_path": self.config["embedding_model_path"]}
                    if self.config["embedding_model_path"]
                    else {}
                ),
            )
            for config in company_configs
        ]
        
        all_agents = seeker_agents + interviewer_agents + company_agents
        for id, agent in enumerate(all_agents):
            agent.set_id(id)

        # Init system prompt for all job agents
        name2company_agent = {agent.name: agent for agent in company_agents}
        for interviewer_agent in interviewer_agents:
            interviewer_agent.init_system_prompt(name2company_agent[interviewer_agent.job.company_name])

        for agent in all_agents:
            memory_config = load_json(
                os.path.join(scene_path, CONFIG_DIR, MEMORY_CONFIG)
            )
            agent.memory = setup_memory(memory_config)
            agent.memory_config = memory_config
        for agent in seeker_agents + interviewer_agents:
            agent.set_embedding(self.embedding_model)

        # assign job_ids_pool for all seekers
        index = faiss.IndexFlatL2(
            self.embedding_model.get_sentence_embedding_dimension()
        )
        index.add(np.array([agent.embedding for agent in interviewer_agents]))
        for seeker_agent in seeker_agents:
            _, job_ids = index.search(
                np.array([seeker_agent.embedding]), self.config["pool_size"]
            )
            seeker_agent.job_ids_pool = list(job_ids[0] + len(seeker_agents))

        self.env.agents = all_agents
        self._set_agent_models()

    def _one_round(self):
        for agent in self.env.agents:
            agent(Msg("system", None, role="system", fun="run"))
        check_pause()

    def run(self):
        play_event.set()

        message_manager.message_queue.put("Start simulation.")
        for r in range(self.cur_round, self.config["round_n"] + 1):
            logger.info(f"Round {r} started")
            self._one_round()
            self.save()
            if stop_event.is_set():
                message_manager.message_queue.put(
                    f"Stop simulation by user at round {r}."
                )
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
