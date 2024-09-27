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
            logger.info(f"Load simulator from {self.config['load_simulator_path']}")
            loaded_simulator = Simulator.load(self.config["load_simulator_path"])
            self.__dict__.update(loaded_simulator.__dict__)
            seeker_agents, interviewer_agents, envs = self._create_agents_envs()

            self.agents = seeker_agents + interviewer_agents

            results = []
            for agent_state, agent in zip(self.agent_save_state, self.agents):
                results.append(agent.load(data=agent_state))
            for res in results:
                res.result()

            self.envs = envs
            self._set_env4agents()            
            logger.info("Load agents and envs successfully")
        else:
            self._init_agents()

        save_configs(self.config)

    def _set_env4agents(self):
        logger.info("Set all_agents for envs")
        agent_dict = {agent.agent_id: agent for agent in self.agents}
        results = []
        for env in self.envs:
            results.append(env.set_attr(attr="all_agents", value=agent_dict))
        for res in tqdm(results, total=len(self.envs), desc="Set all_agents for envs"):
            res.result()
        env = self.envs[0]
        self.env = env

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

    def generate_embedding(self, seeker_configs, interviewer_configs):
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
            seeker_futures = {executor.submit(fetch_seeker_embedding, config): config for config in seeker_configs}
            for future in tqdm(futures.as_completed(seeker_futures), total=len(seeker_futures), desc="Fetching seeker embedding"):
                future.result()
            interviewer_futures = {executor.submit(fetch_interviewer_embedding, config): config for config in interviewer_configs}
            for future in tqdm(futures.as_completed(interviewer_futures), total=len(interviewer_futures), desc="Fetching interviewer embedding"):
                future.result()

    def _create_agents_envs(self, model_configs=None, seeker_configs=None, interviewer_configs=None, memory_config=None):
        if model_configs is None:
            model_configs = load_json(
                os.path.join(scene_path, CONFIG_DIR, self.config["model_configs_path"])
            )
        if seeker_configs is None:
            seeker_configs = load_json(
                os.path.join(scene_path, CONFIG_DIR, self.config['seeker_agent_configs_path'])
            )
        if interviewer_configs is None:
            interviewer_configs = load_json(
                os.path.join(scene_path, CONFIG_DIR, self.config['interviewer_agent_configs_path'])
            )
        if memory_config is None:
            memory_config = load_json(os.path.join(scene_path, CONFIG_DIR, MEMORY_CONFIG))
            memory_config["args"]["embedding_size"] = get_embedding_dimension(self.config["embedding_api"][0])

        llm_num = len(model_configs)
        agent_num = len(seeker_configs) + len(interviewer_configs)
        agent_num_per_llm = math.ceil(agent_num / llm_num)
        embedding_api_num = len(self.config["embedding_api"])
        print("embedding_api_num", embedding_api_num)
        print("embedding api", self.config["embedding_api"])

        # Prepare agent args
        logger.info("Prepare agent args")
        index_ls = list(range(len(seeker_configs + interviewer_configs)))
        random.shuffle(index_ls)
        for config, shuffled_idx in zip(seeker_configs+interviewer_configs, index_ls):
            model_config = model_configs[shuffled_idx//agent_num_per_llm]
            config["args"]["model_config_name"] = model_config["config_name"]
            config["args"]["memory_config"] = memory_config
            config["args"]["embedding_api"] = self.config["embedding_api"][shuffled_idx % embedding_api_num]
        
        # Generate embedding
        logger.info("Generate embedding for agents")
        self.generate_embedding(seeker_configs, interviewer_configs)

        # Init env
        logger.info("Init environment")
        seeker_num_per_env = 200
        env_num = math.ceil(len(seeker_configs) / seeker_num_per_env)
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
            for task in tqdm(futures.as_completed(tasks), total=len(tasks), desc="Init environments"):
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
            for task in tqdm(futures.as_completed(tasks), total=len(tasks), desc="Init seeker agents"):
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
            for task in tqdm(futures.as_completed(tasks), total=len(tasks), desc="Init interviewer agents"):
                interviewer_agents.append(task.result())

        logger.info("searching for job_ids_pool")
        index = faiss.IndexFlatL2(get_embedding_dimension(self.config["embedding_api"][0]))
        index.add(
            np.array([config["args"]["embedding"] for config in interviewer_configs])
        )
        embeddings = np.array([config["args"]["embedding"] for config in seeker_configs])
        _, job_index = index.search(embeddings, self.config["pool_size"])
        for config, index in zip(seeker_configs, job_index):
            config["args"]["job_ids_pool"] = [
                interviewer_agents[i].agent_id for i in list(index)
            ]

        # Just for test
        # for config in seeker_configs:
        #     config["args"]["job_ids_pool"] = [
        #         interviewer_agents[i].agent_id for i in random.sample(range(len(interviewer_agents)), k=self.config["pool_size"])
        #     ]

        logger.info("Set job_ids_pool for seeker agents")
        results = []
        for agent, config in zip(seeker_agents, seeker_configs):
            results.append(agent.set_attr(attr="job_ids_pool", value=config["args"]["job_ids_pool"]))
        for res in tqdm(results, total=len(results), desc="Set job_ids_pool for seeker agents"):
            res.result()

        return seeker_agents, interviewer_agents, envs

    def _init_agents(self):
        # Load configs
        logger.info("Load configs")
        model_configs = load_json(
            os.path.join(scene_path, CONFIG_DIR, self.config["model_configs_path"])
        )
        seeker_configs = load_json(
            os.path.join(scene_path, CONFIG_DIR, self.config['seeker_agent_configs_path'])
        )
        interviewer_configs = load_json(
            os.path.join(scene_path, CONFIG_DIR, self.config['interviewer_agent_configs_path'])
        )
        memory_config = load_json(os.path.join(scene_path, CONFIG_DIR, MEMORY_CONFIG))
        memory_config["args"]["embedding_size"] = get_embedding_dimension(self.config["embedding_api"][0])
        
        seeker_agents, interviewer_agents, envs = self._create_agents_envs(model_configs, seeker_configs, interviewer_configs, memory_config)

        self.agents = seeker_agents + interviewer_agents
        self.envs = envs
        
        self._set_env4agents()

    def _one_round(self):
        results = []
        for agent in self.agents:
            results.append(agent.run())
        for i in range(len(results)):
            results[i] = results[i].result()
        return results

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


    def load(file_path):
        with open(file_path, "rb") as f:
            return dill.load(f)

    def get_save_state(self):
        results = []
        for agent in self.agents:
            results.append(agent.save())
        agent_save_state = []
        for result in results:
            agent_save_state.append(result.result())
        
        return agent_save_state

    def save(self):
        try:
            file_manager = FileManager.get_instance()
            save_path = os.path.join(file_manager.run_dir, f"ROUND-{self.cur_round}.pkl")
            self.agent_save_state = self.get_save_state()
            self.cur_round += 1
            with open(save_path, "wb") as f:
                dill.dump(self, f)
            logger.info(f"Saved simulator to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save simulator: {e}")


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
