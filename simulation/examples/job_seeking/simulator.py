import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import dill
from loguru import logger
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

import agentscope
from agentscope.file_manager import file_manager
from agentscope.message import Msg
from agentscope.agents.agent import DistConf

from simulation.helpers.simulator import BaseSimulator
from simulation.helpers.events import (
    play_event,
    stop_event,
    pause_success_event,
    check_pause,
)
from simulation.helpers.message import message_manager
from simulation.helpers.constants import *
from agentscope.constants import _DEFAULT_DIR
from simulation.examples.job_seeking.agent import *
from simulation.helpers.emb_service import *
from simulation.helpers.utils import *

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
                self.config["save_dir"] if self.config["save_dir"] else _DEFAULT_DIR
            ),
        )

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            return dill.load(f)

    def _init_agents(self):
        # Load configs
        seeker_configs = load_json(
            os.path.join(scene_path, CONFIG_DIR, SEEKER_AGENT_CONFIG)
        )
        interviewer_configs = load_json(
            os.path.join(scene_path, CONFIG_DIR, INTERVIEWER_AGENT_CONFIG)
        )
        company_configs = load_json(
            os.path.join(scene_path, CONFIG_DIR, COMPANY_AGENT_CONFIG)
        )
        memory_config = load_json(os.path.join(scene_path, CONFIG_DIR, MEMORY_CONFIG))

        # Prepare agent args
        for config in seeker_configs + interviewer_configs + company_configs:
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

        for config in company_configs:
            name, cd = config["args"]["name"], config["args"]["cd"]
            config["args"]["embedding"] = get_embedding(
                f"{name} {cd}", self.config["embedding_api"]
            )

        # Init agents
        env_agent = EnvironmentAgent(
            name="environment",
            to_dist=DistConf(host="localhost", port=self.config["env_agent_port"]),
        )

        seeker_agents = [
            SeekerAgent(
                env_agent=env_agent,
                **config["args"],
                to_dist=DistConf(
                    host=config["args"]["host"], port=config["args"]["port"]
                ),
            )
            for config in seeker_configs
        ]
        interviewer_agents = [
            InterviewerAgent(
                env_agent=env_agent,
                **config["args"],
                to_dist=DistConf(
                    host=config["args"]["host"], port=config["args"]["port"]
                ),
            )
            for config in interviewer_configs
        ]
        company_agents = [
            CompanyAgent(
                env_agent=env_agent,
                **config["args"],
                to_dist=DistConf(
                    host=config["args"]["host"], port=config["args"]["port"]
                ),
            )
            for config in company_configs
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

        results = []
        for agent, config in zip(seeker_agents, seeker_configs):
            results.append(
                agent(
                    get_assistant_msg(
                        fun="set_attr",
                        params={
                            "attr": "job_ids_pool",
                            "value": config["args"]["job_ids_pool"],
                        },
                    )
                )
            )
        for res in results:
            res["content"]

        agent_distribution_infos = {}
        for agent in seeker_agents + interviewer_agents + company_agents:
            agent_distribution_infos[agent.agent_id] = {
                "host": agent.host,
                "port": agent.port,
                "agent_id": agent.agent_id,
            }
        env_agent(
            get_assistant_msg(
                fun="set_agent_distribution_infos",
                params={"agent_distribution_infos": agent_distribution_infos},
            )
        )["content"]

        self.agents = seeker_agents + interviewer_agents + company_agents + [env_agent]

    def get_agent_by_id(self, agent_id: str):
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None

    def _one_round(self):
        results = []
        for agent in self.agents:
            results.append(agent(Msg("system", None, role="system", fun="run")))

        for res in results:
            print(res["content"])

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
