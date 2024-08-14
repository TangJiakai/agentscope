import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import dill
from loguru import logger
import numpy as np
import faiss

import agentscope
from agentscope.manager import FileManager
from agentscope.agents.agent import DistConf

from simulation.helpers.events import (
    play_event,
    stop_event,
    kill_event,
    pause_success_event,
    check_pause,
)
from simulation.helpers.message import message_manager
from simulation.helpers.constants import *
from agentscope.constants import _DEFAULT_SAVE_DIR
from simulation.examples.job_seeking.agent import *
from simulation.examples.job_seeking.env import JobSeekingEnv
from simulation.helpers.emb_service import *
from simulation.helpers.utils import *

CUR_ROUND = 1
SEEKER_AGENT_CONFIG = "seeker_agent_configs.json"
INTERVIEWER_AGENT_CONFIG = "interviewer_agent_configs.json"

scene_path = os.path.dirname(os.path.abspath(__file__))


class Simulator:
    def __init__(self):
        super().__init__()
        self.config = load_yaml(os.path.join(scene_path, CONFIG_DIR, SIMULATION_CONFIG))

        global CUR_ROUND
        self.cur_round = CUR_ROUND
        self._from_scratch()

    def _from_scratch(self):
        self._init_agentscope()

        if self.config["load_simulator_path"] is not None:
            loaded_simulator = Simulator.load(self.config["load_simulator_path"])
            self.__dict__.update(loaded_simulator.__dict__)
            global CUR_ROUND
            CUR_ROUND = self.cur_round
        else:
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
            save_dir=(
                self.config["save_dir"]
                if self.config["save_dir"]
                else _DEFAULT_SAVE_DIR
            ),
        )

    def _init_agents(self):
        # Load configs
        seeker_configs = load_json(
            os.path.join(scene_path, CONFIG_DIR, SEEKER_AGENT_CONFIG)
        )
        interviewer_configs = load_json(
            os.path.join(scene_path, CONFIG_DIR, INTERVIEWER_AGENT_CONFIG)
        )
        memory_config = load_json(os.path.join(scene_path, CONFIG_DIR, MEMORY_CONFIG))

        # Prepare agent args
        for config in seeker_configs + interviewer_configs:
            memory_config["args"]["embedding_size"] = get_embedding_dimension(
                self.config["embedding_api"]
            )
            config["args"]["memory_config"] = memory_config
            config["args"]["embedding_api"] = self.config["embedding_api"]

        for config in seeker_configs:
            cv = str(config["args"]["cv"])
            config["args"]["embedding"] = get_embedding(
                cv, self.config["embedding_api"]
            )

        for config in interviewer_configs:
            name, jd, jr = (
                config["args"]["name"],
                config["args"]["jd"],
                config["args"]["jr"],
            )
            config["args"]["embedding"] = get_embedding(
                f"{name} {jd} {' '.join(jr)}", self.config["embedding_api"]
            )

        # Init env
        env = JobSeekingEnv(
            name="environment",
            to_dist=DistConf(host=self.config["host"], port=self.config["base_port"]),
        )

        # Init agents
        seeker_agents = [
            SeekerAgent(
                env=env,
                **config["args"],
                to_dist=DistConf(
                    host=config["args"]["host"], port=config["args"]["port"]
                ),
            )
            for config in seeker_configs
        ]
        interviewer_agents = [
            InterviewerAgent(
                env=env,
                **config["args"],
                to_dist=DistConf(
                    host=config["args"]["host"], port=config["args"]["port"]
                ),
            )
            for config in interviewer_configs
        ]

        index = faiss.IndexFlatL2(get_embedding_dimension(self.config["embedding_api"]))
        index.add(
            np.array([config["args"]["embedding"] for config in interviewer_configs])
        )
        for config in seeker_configs:
            _, job_index = index.search(
                np.array([config["args"]["embedding"]]), self.config["pool_size"]
            )
            config["args"]["job_ids_pool"] = [
                interviewer_agents[index].agent_id for index in list(job_index[0])
            ]

        for agent, config in zip(seeker_agents, seeker_configs):
            agent.set_attr(attr="job_ids_pool", value=config["args"]["job_ids_pool"])

        env.set_attr(attr="all_agents", value=seeker_agents + interviewer_agents)

        self.agents = seeker_agents + interviewer_agents

    def get_agent_by_id(self, agent_id: str):
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None

    def _one_round(self):
        results = []
        for agent in self.agents:
            results.append(agent.run())
        for res in results:
            print(res.get())

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
            pause_success_event.set()
            check_pause()
            if kill_event.is_set():
                logger.info(f"Kill simulation by user at round {r}.")
                return
        message_manager.message_queue.put("Simulation finished.")
        logger.info("Simulation finished")

    def load(file_path):
        with open(file_path, "rb") as f:
            return dill.load(f)

    def save(self):
        file_manager = FileManager.get_instance()
        save_path = os.path.join(file_manager.run_dir, f"ROUND-{self.cur_round}.pkl")
        global CUR_ROUND
        self.cur_round = CUR_ROUND + 1
        CUR_ROUND = self.cur_round
        with open(save_path, "wb") as f:
            dill.dump(self, f)
        logger.info(f"Saved simulator to {save_path}")


if __name__ == "__main__":
    simulator = Simulator()
    simulator.run()
