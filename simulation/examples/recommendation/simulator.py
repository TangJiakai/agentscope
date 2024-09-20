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
from simulation.examples.recommendation.agent import *
from simulation.examples.recommendation.environment.env import RecommendationEnv
from simulation.helpers.emb_service import *
from simulation.helpers.utils import *

CUR_ROUND = 1

scene_path = os.path.dirname(os.path.abspath(__file__))


class Simulator:
    def __init__(self):
        super().__init__()
        self.config = load_yaml(os.path.join(scene_path, CONFIG_DIR, SIMULATION_CONFIG))

        self.cur_round = 1
        self._from_scratch()

    def _from_scratch(self):
        self._init_agentscope()

        if self.config["load_simulator_path"] is not None:
            loaded_simulator = Simulator.load(self.config["load_simulator_path"])
            self.__dict__.update(loaded_simulator.__dict__)
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
        logger.info("Load configs")
        model_configs = load_json(
            os.path.join(scene_path, CONFIG_DIR, self.config["model_configs_path"])
        )
        agent_configs = load_json(os.path.join(scene_path, CONFIG_DIR, self.config["recuser_agent_configs_path"]))
        memory_config = load_json(os.path.join(scene_path, CONFIG_DIR, MEMORY_CONFIG))
        item_infos = load_json(os.path.join(scene_path, CONFIG_DIR, self.config['item_infos_path']))

        llm_num = len(model_configs)
        agent_num = len(agent_configs)
        agent_num_per_llm = math.ceil(agent_num / llm_num)

        # Prepare agent args
        logger.info("Prepare agent args")
        index_ls = list(range(agent_num))
        agent_relationships = []
        random.shuffle(index_ls)
        for config, shuffled_idx in zip(agent_configs, index_ls):
            model_config = model_configs[shuffled_idx//agent_num_per_llm]
            config["args"]["model_config_name"] = model_config["config_name"]
            memory_config["args"]["embedding_size"] = get_embedding_dimension(
                self.config["embedding_api"]
            )
            config["args"]["memory_config"] = memory_config
            config["args"]["embedding_api"] = self.config["embedding_api"]
            agent_relationships.append(config["args"].pop("relationship"))

        # Init env
        logger.info("Init environment")
        user_num_per_env = 200
        env_num = math.ceil(len(agent_configs) / user_num_per_env)
        env_names = [f"environment-{str(i)}" for i in range(env_num)]
        env_ports = [i % self.config["server_num_per_host"] + self.config["base_port"] for i in range(env_num)]

        envs = []
        tasks = []
        with futures.ThreadPoolExecutor() as executor:
            for name, port in zip(env_names, env_ports):
                tasks.append(
                    executor.submit(
                        RecommendationEnv,
                        name=name,
                        item_infos=item_infos,
                        embedding_api=self.config["embedding_api"],
                        to_dist=DistConf(
                            host=config["args"]["host"], port=port
                        ),
                    ),
                )
            for task in tasks:
                envs.append(task.result())

        # Init agents
        logger.info(f"Init {len(agent_configs)} recuser agents")
        agents = []
        tasks = []
        with futures.ThreadPoolExecutor() as executor:
            for i, config in enumerate(agent_configs):
                tasks.append(
                    executor.submit(
                        RecUserAgent,
                        env=envs[i // agent_num_per_llm],
                        **config["args"],
                        to_dist=DistConf(
                            host=config["args"]["host"], port=config["args"]["port"]
                        ),
                    ),
                )
            for task in tasks:
                agents.append(task.result())

        logger.info("Set relationship to agents")
        results = []
        for i, agent in enumerate(agents):
            results.append(agent.set_attr(
                "relationship",
                {agents[j].agent_id: agents[j] for j in agent_relationships[i]},
            ))
        for res in results:
            res.result()

        logger.info("Set all agents to envs")  
        results = []
        for env in envs:
            results.append(env.set_attr(attr="all_agents", value={agent.agent_id: agent for agent in agents}))
        for res in results:
            res.result()

        self.agents = agents
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
        #     "http://localhost:9111/store_message",
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
        self.cur_round += 1
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
