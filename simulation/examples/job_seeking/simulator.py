from datetime import timedelta
import math
import os
import random
import sys

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import dill
import time
from concurrent import futures
from loguru import logger
import numpy as np
import faiss

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
from simulation.examples.job_seeking.agent import *
from simulation.helpers.base_env import BaseEnv
from simulation.helpers.emb_service import *
from simulation.helpers.utils import *
from simulation.helpers.base_simulator import BaseSimulator

CUR_ROUND = 1

scene_path = os.path.dirname(os.path.abspath(__file__))


class Simulator(BaseSimulator):
    def __init__(self):
        super().__init__(scene_path=scene_path)

    def generate_embedding(self, seeker_configs, interviewer_configs):
        logger.info("Generate embedding for agents")

        def fetch_seeker_embedding(config):
            cv = str(config["args"]["cv"])
            config["args"]["embedding"] = get_embedding(
                cv, config["args"]["embedding_api"]
            )
            return config

        def fetch_interviewer_embedding(config):
            name, jd, jr = (
                config["args"]["name"],
                config["args"]["jd"],
                config["args"]["jr"],
            )
            config["args"]["embedding"] = get_embedding(
                f"{name} {jd} {' '.join(jr)}", config["args"]["embedding_api"]
            )
            return config

        with futures.ThreadPoolExecutor() as executor:
            for _ in tqdm(
                executor.map(fetch_seeker_embedding, seeker_configs),
                total=len(seeker_configs),
                desc="Fetching seeker embedding",
            ):
                pass

            for _ in tqdm(
                executor.map(fetch_interviewer_embedding, interviewer_configs),
                total=len(interviewer_configs),
                desc="Fetching interviewer embedding",
            ):
                pass

    def search_for_job_ids_pool_gpu(
        self, seeker_configs, interviewer_configs, interviewer_agents
    ):
        d = get_embedding_dimension(self.config["embedding_api"][0])
        nlist = 100
        gpu_res = faiss.StandardGpuResources()
        batch_size = 10000
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        gpu_index = faiss.index_cpu_to_gpu(gpu_res, self.config["gpu_id"], index)

        interviewer_embeddings = np.array(
            [config["args"]["embedding"] for config in interviewer_configs]
        )
        gpu_index.train(interviewer_embeddings)

        num_interviewer_batches = (
            len(interviewer_embeddings) + batch_size - 1
        ) // batch_size

        for i in tqdm(
            range(num_interviewer_batches),
            total=num_interviewer_batches,
            desc="Adding interviewer embeddings",
        ):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(interviewer_embeddings))
            gpu_index.add(interviewer_embeddings[start_idx:end_idx])

        seeker_embeddings = np.array(
            [config["args"]["embedding"] for config in seeker_configs]
        )
        _, job_index = [], []
        num_seeker_batches = (len(seeker_embeddings) + batch_size - 1) // batch_size
        for i in tqdm(
            range(num_seeker_batches),
            total=num_seeker_batches,
            desc="Searching for job_ids_pool",
        ):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(seeker_embeddings))
            _, batch_job_index = gpu_index.search(
                seeker_embeddings[start_idx:end_idx], self.config["pool_size"]
            )
            job_index.extend(batch_job_index)

        for config, idx in zip(seeker_configs, job_index):
            config["args"]["job_ids_pool"] = [
                interviewer_agents[i].agent_id for i in list(idx)
            ]

    def search_for_job_ids_pool_cpu(
        self, seeker_configs, interviewer_configs, interviewer_agents
    ):
        index = faiss.IndexFlatL2(
            get_embedding_dimension(self.config["embedding_api"][0])
        )
        index.add(
            np.array([config["args"]["embedding"] for config in interviewer_configs])
        )
        embeddings = np.array(
            [config["args"]["embedding"] for config in seeker_configs]
        )
        _, job_index = index.search(embeddings, self.config["pool_size"])
        for config, index in zip(seeker_configs, job_index):
            config["args"]["job_ids_pool"] = [
                interviewer_agents[i].agent_id for i in list(index)
            ]

    def search_for_job_ids_pool(
        self, seeker_configs, interviewer_configs, interviewer_agents
    ):
        self.search_for_job_ids_pool_cpu(
            seeker_configs, interviewer_configs, interviewer_agents
        )

    def set_job_ids_pool(self, seeker_agents, seeker_configs):
        logger.info("Set job_ids_pool for seeker agents")
        results = []
        for agent, config in zip(seeker_agents, seeker_configs):
            results.append(
                agent.set_attr(
                    attr="job_ids_pool", value=config["args"]["job_ids_pool"]
                )
            )
        for res in tqdm(
            results, total=len(results), desc="Set job_ids_pool for seeker agents"
        ):
            res.result()

    def _prepare_agents_args(self):
        logger.info("Load configs")
        memory_config = load_json(os.path.join(scene_path, CONFIG_DIR, MEMORY_CONFIG))
        model_configs = load_json(os.path.join(scene_path, CONFIG_DIR, MODEL_CONFIG))
        seeker_configs = load_json(
            os.path.join(
                scene_path, CONFIG_DIR, self.config["seeker_agent_configs_path"]
            )
        )
        interviewer_configs = load_json(
            os.path.join(
                scene_path, CONFIG_DIR, self.config["interviewer_agent_configs_path"]
            )
        )

        logger.info("Prepare agents args")
        llm_num = len(model_configs)
        agent_num = len(seeker_configs) + len(interviewer_configs)
        agent_num_per_llm = math.ceil(agent_num / llm_num)
        embedding_api_num = len(self.config["embedding_api"])
        logger.info(f"llm_num: {llm_num}")
        logger.info(f"agent_num: {agent_num}")
        logger.info(f"agent_num_per_llm: {agent_num_per_llm}")
        logger.info(f"embedding_api_num: {embedding_api_num}")
        memory_config["args"]["embedding_size"] = get_embedding_dimension(
            self.config["embedding_api"][0]
        )

        # Prepare agent args
        logger.info("Prepare agent args")
        index_ls = list(range(len(seeker_configs + interviewer_configs)))
        random.shuffle(index_ls)
        for config, shuffled_idx in zip(seeker_configs + interviewer_configs, index_ls):
            model_config = model_configs[shuffled_idx // agent_num_per_llm]
            config["args"]["model_config_name"] = model_config["config_name"]
            config["args"]["memory_config"] = None if self.resume else memory_config
            config["args"]["embedding_api"] = self.config["embedding_api"][
                shuffled_idx % embedding_api_num
            ]

        return seeker_configs, interviewer_configs

    def _create_envs(self, agent_num):
        logger.info("Init environment")
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
                    "to_dist": DistConf(host=self.config["host"], port=port),
                }
                for name, port in zip(env_names, env_ports)
            ]
            for env in tqdm(
                executor.map(lambda arg: BaseEnv(**arg), args),
                total=len(args),
                desc="Init environments",
            ):
                envs.append(env)

        self.envs = envs
        self.env = envs[0]

    def _create_agents(self, seeker_configs, interviewer_configs):
        logger.info(f"Init {len(seeker_configs)} seeker agents")
        seeker_agents = []
        envs = self.envs
        env_num = len(envs)
        with futures.ThreadPoolExecutor() as executor:
            args = [
                {
                    "env": envs[i % env_num],
                    **config["args"],
                    "to_dist": DistConf(
                        host=config["args"]["host"], port=config["args"]["port"]
                    ),
                }
                for i, config in enumerate(seeker_configs)
            ]
            for agent in tqdm(
                executor.map(lambda arg: SeekerAgent(**arg), args),
                total=len(seeker_configs),
                desc="Init seeker agents",
            ):
                seeker_agents.append(agent)

        logger.info(f"Init {len(interviewer_configs)} interviewer agents")
        interviewer_agents = []
        with futures.ThreadPoolExecutor() as executor:
            args = [
                {
                    "env": envs[i % env_num],
                    **config["args"],
                    "to_dist": DistConf(
                        host=config["args"]["host"], port=config["args"]["port"]
                    ),
                }
                for i, config in enumerate(interviewer_configs)
            ]
            for agent in tqdm(
                executor.map(lambda arg: InterviewerAgent(**arg), args),
                total=len(interviewer_configs),
                desc="Init interviewer agents",
            ):
                interviewer_agents.append(agent)
        self.agents = seeker_agents + interviewer_agents
        return seeker_agents, interviewer_agents

    def _init_agents_envs(self):
        # Prepare agents args
        seeker_configs, interviewer_configs = self._prepare_agents_args()
        # Init envs
        self._create_envs(len(seeker_configs) + len(interviewer_configs))
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
        seeker_agents, interviewer_agents = self._create_agents(
            seeker_configs, interviewer_configs
        )
        # Resume agents
        if self.resume:
            logger.info("Resume agents...")
            results = []
            for agent, state in zip(
                seeker_agents + interviewer_agents, self.agent_save_state
            ):
                results.append(agent.load(state))
                agent.agent_id = dill.loads(state)["_oid"]
            for res in tqdm(results, total=len(results), desc="Resume agents"):
                res.result()

        # Set all_agents for envs
        self._set_env4agents()

        # Generate embedding and search for job_ids_pool
        if not self.resume:
            self.generate_embedding(seeker_configs, interviewer_configs)
            self.search_for_job_ids_pool(
                seeker_configs, interviewer_configs, interviewer_agents
            )
            self.set_job_ids_pool(seeker_agents, seeker_configs)

    def run(self):
        play_event.set()

        message_manager.message_queue.put("Start simulation.")
        for r in range(self.cur_round, self.config["round_n"] + 1):
            logger.info(f"Round {r} started")
            results = self._one_round()
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

            no_seeking_cnt = 0
            no_job_cnt = 0
            get_job_cnt = 0
            for result in results:
                if result == "No Seeking Job.":
                    no_seeking_cnt += 1
                elif result == -1:
                    no_job_cnt += 1
                elif result == "Done":
                    pass
                else:
                    get_job_cnt += 1

            logger.info("====================================")
            logger.info(f"Round {r} finished")
            logger.info(f"No Seeking Job: {no_seeking_cnt}")
            logger.info(f"No Job: {no_job_cnt}")
            logger.info(f"Get Job: {get_job_cnt}")
            logger.info("====================================")

            # message_save_path = "/data/tangjiakai/general_simulation/"
            # resp = requests.post(
            #     "http://localhost:9111/store_message",
            #     json={
            #         "save_data_path": os.path.join(message_save_path, f"Round-{r}.json"),
            #     }
            # )

        message_manager.message_queue.put("Simulation finished.")
        logger.info("Simulation finished")

    def get_save_state(self):
        results = []
        for agent in self.agents:
            results.append(agent.save())
        agent_save_state = []
        for result in tqdm(results, total=len(results), desc="Get agent save state"):
            agent_save_state.append(result.result())

        results = []
        for env in self.envs:
            results.append(env.save())
        env_save_state = []
        for result in tqdm(results, total=len(results), desc="Get env save state"):
            env_save_state.append(result.result())

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
