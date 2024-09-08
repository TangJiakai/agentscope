from datetime import timedelta
import math
import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import dill
import time
from concurrent import futures
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
from simulation.helpers.base_env import BaseEnv
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
        model_configs = load_json(
            os.path.join(scene_path, CONFIG_DIR, self.config["model_configs_path"])
        )
        seeker_configs = load_json(
            os.path.join(scene_path, CONFIG_DIR, SEEKER_AGENT_CONFIG)
        )
        interviewer_configs = load_json(
            os.path.join(scene_path, CONFIG_DIR, INTERVIEWER_AGENT_CONFIG)
        )
        memory_config = load_json(os.path.join(scene_path, CONFIG_DIR, MEMORY_CONFIG))
        memory_config["args"]["embedding_size"] = get_embedding_dimension(self.config["embedding_api"])
        memory_config["args"]
        
        llm_num = len(model_configs)
        agent_num = len(seeker_configs) + len(interviewer_configs)
        agent_num_per_llm = math.ceil(agent_num / llm_num)

        # Prepare agent args
        index_ls = list(range(len(seeker_configs + interviewer_configs)))
        random.shuffle(index_ls)
        for config, shuffled_idx in zip(seeker_configs+interviewer_configs, index_ls):
            model_config = model_configs[shuffled_idx//agent_num_per_llm]
            config["args"]["model_config_name"] = model_config["config_name"]
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
        logger.info("Init environment")
        seeker_num_per_env = 200
        env_num = (len(seeker_configs) + seeker_num_per_env - 1) // seeker_num_per_env
        env_names = [f"environment-{str(i)}" for i in range(env_num)]
        env_ports = [i % self.config["server_num_per_host"] + self.config["base_port"] for i in range(env_num)]

        envs = []
        tasks = []
        with futures.ThreadPoolExecutor() as executor:
            for name, port in zip(env_names, env_ports):
                tasks.append(
                    executor.submit(
                        BaseEnv,
                        name=name,
                        to_dist=DistConf(
                            host=config["args"]["host"], port=port
                        ),
                    ),
                )
            for task in tasks:
                envs.append(task.result())

        # Init agents
        logger.info(f"Init {len(seeker_configs)} seeker agents")
        seeker_agents = []
        tasks = []
        with futures.ThreadPoolExecutor() as executor:
            for i, config in enumerate(seeker_configs):
                tasks.append(
                    executor.submit(
                        SeekerAgent,
                        env=envs[i // seeker_num_per_env],
                        **config["args"],
                        to_dist=DistConf(
                            host=config["args"]["host"], port=config["args"]["port"]
                        ),
                    ),
                )
            for task in tasks:
                seeker_agents.append(task.result())
        
        logger.info(f"Init {len(interviewer_configs)} interviewer agents")
        interviewer_agents = []
        tasks = []
        with futures.ThreadPoolExecutor() as executor:
            for config in interviewer_configs:
                tasks.append(
                    executor.submit(
                        InterviewerAgent,
                        env=None,
                        **config["args"],
                        to_dist=DistConf(
                            host=config["args"]["host"], port=config["args"]["port"]
                        ),
                    ),
                )
            for task in tasks:
                interviewer_agents.append(task.result())

        index = faiss.IndexFlatL2(get_embedding_dimension(self.config["embedding_api"]))
        index.add(
            np.array([config["args"]["embedding"] for config in interviewer_configs])
        )

        tasks = []
        with futures.ThreadPoolExecutor() as executor:
            for config in seeker_configs:
                tasks.append(
                    executor.submit(
                        index.search,
                        np.array([config["args"]["embedding"]]),
                        self.config["pool_size"],
                    ),
                )
            for config, task in zip(seeker_configs, tasks):
                _, job_index = task.result()
                config["args"]["job_ids_pool"] = [
                    interviewer_agents[index].agent_id for index in list(job_index[0])
                ]

        results = []
        for agent, config in zip(seeker_agents, seeker_configs):
            results.append(agent.set_attr(attr="job_ids_pool", value=config["args"]["job_ids_pool"]))
        for res in results:
            res.result()

        agent_dict = {agent.agent_id: agent for agent in seeker_agents + interviewer_agents}
        results = []
        for env in envs:
            results.append(env.set_attr(attr="all_agents", value=agent_dict))
        for res in results:
            res.result()

        self.agents = seeker_agents + interviewer_agents
        self.envs = envs
        self.env = envs[0]

    def _one_round(self):
        results = []
        for agent in self.agents:
            results.append(agent.run())
        for res in results:
            print(res.result())

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

        # message_save_path = "/data/tangjiakai/general_simulation/tmp_message.json"
        # resp = requests.post(
        #     "http://localhost:9000/store_message",
        #     json={
        #         "save_data_path": message_save_path,
        #     }
        # )

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
    start_time = time.time()
    simulator = Simulator()
    end_time = time.time()
    formatted_time = str(timedelta(seconds=end_time - start_time))
    logger.info(f"Init Agent Total time: {formatted_time}")

    start_time = time.time()
    simulator.run()
    end_time = time.time()
    formatted_time = str(timedelta(seconds=end_time - start_time))
    logger.info(f"Simulation Total time: {formatted_time}")
