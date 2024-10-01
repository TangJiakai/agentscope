from datetime import timedelta
import math
import os
import random
import sys
import faiss
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import dill
import time
from concurrent import futures
from loguru import logger

import agentscope
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
from simulation.helpers.base_simulator import BaseSimulator

CUR_ROUND = 1

scene_path = os.path.dirname(os.path.abspath(__file__))


class Simulator(BaseSimulator):
    def __init__(self):
        super().__init__(scene_path=scene_path)

    def _init_agentscope(self):
        agentscope.init(
            project=self.config["project_name"],
            save_code=False,
            save_api_invoke=False,
            model_configs=os.path.join(
                scene_path, CONFIG_DIR, self.config["model_configs_path"]
            ),
            use_monitor=False,
            save_dir=os.path.join(_DEFAULT_SAVE_DIR, self.config["project_name"]),
            runtime_id=self.config["runtime_id"],
        )

    def _prepare_agents_args(self):
        logger.info("Load configs")
        memory_config = load_json(os.path.join(scene_path, CONFIG_DIR, MEMORY_CONFIG))
        model_configs = load_json(os.path.join(scene_path, CONFIG_DIR, MODEL_CONFIG))
        agent_configs = load_json(
            os.path.join(
                scene_path, CONFIG_DIR, self.config["recuser_agent_configs_path"]
            )
        )

        logger.info("Prepare agents args")
        llm_num = len(model_configs)
        agent_num = len(agent_configs)
        agent_num_per_llm = math.ceil(agent_num / llm_num)
        embedding_api_num = len(self.config["embedding_api"])
        logger.info(f"llm_num: {llm_num}")
        logger.info(f"agent_num: {agent_num}")
        logger.info(f"agent_num_per_llm: {agent_num_per_llm}")
        logger.info(f"embedding_api_num: {embedding_api_num}")
        memory_config["args"]["embedding_size"] = get_embedding_dimension(
            self.config["embedding_api"][0]
        )

        index_ls = list(range(agent_num))
        agent_relationships = []
        random.shuffle(index_ls)
        for config, shuffled_idx in zip(agent_configs, index_ls):
            model_config = model_configs[shuffled_idx // agent_num_per_llm]
            config["args"]["model_config_name"] = model_config["config_name"]
            memory_config["args"]["embedding_size"] = get_embedding_dimension(
                self.config["embedding_api"][0]
            )
            config["args"]["memory_config"] = None if self.resume else memory_config
            config["args"]["embedding_api"] = self.config["embedding_api"][
                shuffled_idx % embedding_api_num
            ]
            agent_relationships.append(config["args"].pop("relationship"))

        return agent_configs, agent_relationships

    def _create_envs(self, agent_num):
        # Load Item Infos
        logger.info("Load Item Infos")
        item_infos = load_json(
            os.path.join(scene_path, CONFIG_DIR, self.config["item_infos_path"])
        )

        # Init Index
        index = None
        if not self.resume:
            logger.info("Init Index")
            item_embs = []
            with futures.ThreadPoolExecutor() as executor:
                args = [
                    {
                        "sentence": item_info["title"] + item_info["genres"],
                        "api": self.config["embedding_api"][0],
                    }
                    for item_info in item_infos
                ]
                for item_emb in tqdm(
                    executor.map(lambda arg: get_embedding(**arg), args),
                    total=len(item_infos),
                    desc="Init Index",
                ):
                    item_embs.append(item_emb)

            index = faiss.IndexFlatL2(
                get_embedding_dimension(self.config["embedding_api"][0])
            )
            index.add(np.array(item_embs))

        logger.info("Init environment")
        embedding_api_num = len(self.config["embedding_api"])
        env_num = math.ceil(agent_num / AGENT_PER_ENV)
        env_names = [f"environment-{str(i)}" for i in range(env_num)]
        env_ports = [
            i % self.config["server_num_per_host"] + self.config["base_port"]
            for i in range(env_num)
        ]
        envs = []
        with futures.ThreadPoolExecutor() as executor:
            args = [
                {
                    "name": name,
                    "embedding_api": self.config["embedding_api"][
                        i % embedding_api_num
                    ],
                    "item_infos": item_infos,
                    "index": faiss.serialize_index(index) if not self.resume else None,
                    "to_dist": DistConf(host=self.config["host"], port=port),
                }
                for i, name, port in zip(range(len(env_names)), env_names, env_ports)
            ]
            for env in tqdm(
                executor.map(lambda arg: RecommendationEnv(**arg), args),
                total=len(env_names),
                desc="Init environments",
            ):
                envs.append(env)

        self.envs = envs
        self.env = envs[0]

    def _create_agents(self, agent_configs, agent_relationships):
        logger.info(f"Init {len(agent_configs)} recuser agents")
        env_num = len(self.envs)
        agents = []
        with futures.ThreadPoolExecutor() as executor:
            args = [
                {
                    "env": self.envs[i % env_num],
                    **config["args"],
                    "to_dist": DistConf(
                        host=config["args"]["host"], port=config["args"]["port"]
                    ),
                }
                for i, config in enumerate(agent_configs)
            ]
            for agent in tqdm(
                executor.map(lambda arg: RecUserAgent(**arg), args),
                total=len(agent_configs),
                desc="Init agents",
            ):
                agents.append(agent)

        logger.info("Set relationship to agents")
        results = []
        for i, agent in enumerate(agents):
            results.append(
                agent.set_attr(
                    "relationship",
                    {agents[j].agent_id: agents[j] for j in agent_relationships[i]},
                )
            )
        for res in tqdm(results, total=len(results), desc="Set relationship to agents"):
            res.result()
        self.agents = agents
        return agents

    def _init_agents_envs(self):
        # Prepare agents args
        agent_configs, agent_relationships = self._prepare_agents_args()
        # Init envs
        self._create_envs(len(agent_configs))
        # Resume envs
        if self.resume:
            logger.info("Resume envs...")
            results = []
            for env, state in zip(self.envs, self.env_save_state):
                results.append(env.load(state))
                env.agent_id = dill.loads(state)["_oid"]
            for res in tqdm(results, total=len(results), desc="Resume envs"):
                res.result()

        # Init agents
        agents = self._create_agents(agent_configs, agent_relationships)
        # Resume agents
        if self.resume:
            logger.info("Resume agents...")
            results = []
            for agent, state in zip(agents, self.agent_save_state):
                results.append(agent.load(state))
                agent.agent_id = dill.loads(state)["_oid"]
            for res in tqdm(results, total=len(results), desc="Resume agents"):
                res.result()

        # Set all_agents for envs
        self._set_env4agents()

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

            message_save_path = os.path.join(
                "/mnt/jiakai/GeneralSimulation/runs",
                self.config["project_name"],
                self.config["runtime_id"],
            )
            resp = requests.post(
                "http://localhost:9111/store_message",
                json={
                    "save_data_path": os.path.join(
                        message_save_path, f"Round-{r}.json"
                    ),
                },
            )

        message_manager.message_queue.put("Simulation finished.")
        logger.info("Simulation finished")

    def load(file_path):
        with open(file_path, "rb") as f:
            return dill.load(f)

    def get_save_state(self):
        results = []
        for agent in self.agents:
            results.append(agent.save())
        agent_save_state = []
        for res in tqdm(results, total=len(results), desc="Get agent save state"):
            agent_save_state.append(res.result())

        results = []
        for env in self.envs:
            results.append(env.save())
        env_save_state = []
        for res in tqdm(results, total=len(results), desc="Get env save state"):
            env_save_state.append(res.result())

        return agent_save_state, env_save_state


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
