from datetime import timedelta
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import dill
import time
from concurrent import futures
from loguru import logger

import agentscope
from agentscope.manager import FileManager
from agentscope.message import Msg
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
AGENT_CONFIG = "recuser_agent_configs.json"
ITEM_INFOS = "item_infos.json"

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
        agent_configs = load_json(os.path.join(scene_path, CONFIG_DIR, AGENT_CONFIG))
        memory_config = load_json(os.path.join(scene_path, CONFIG_DIR, MEMORY_CONFIG))
        item_infos = load_json(os.path.join(scene_path, CONFIG_DIR, ITEM_INFOS))

        # Prepare agent args
        agent_relationships = []
        for config in agent_configs:
            memory_config["args"]["embedding_size"] = get_embedding_dimension(
                self.config["embedding_api"]
            )
            config["args"]["memory_config"] = memory_config
            config["args"]["embedding_api"] = self.config["embedding_api"]
            agent_relationships.append(config["args"].pop("relationship"))

        for config in agent_configs:
            interest = str(config["args"]["interest"])
            config["args"]["embedding"] = get_embedding(
                interest, self.config["embedding_api"]
            )
        
        # Init env
        logger.info("Init environment")
        env = RecommendationEnv(
            name="env",
            item_infos=item_infos,
            embedding_api=self.config["embedding_api"],
            to_dist=DistConf(host=self.config["host"], port=self.config["base_port"]),
        )

        # Init agents
        ist = time.time()
        logger.info(f"Init {len(agent_configs)} recuser agents")
        agents = []
        tasks = []
        with futures.ThreadPoolExecutor() as executor:
            for config in agent_configs:
                tasks.append(
                    executor.submit(
                        RecUserAgent,
                        env=env,
                        **config["args"],
                        to_dist=DistConf(
                            host=config["args"]["host"], port=config["args"]["port"]
                        ),
                    ),
                )
            for task in tasks:
                agents.append(task.result())
        iet = time.time()
        logger.info(f"Init agents time: {iet - ist:.2f}s")

        results = []
        for i, agent in enumerate(agents):
            results.append(agent.set_attr(
                "relationship",
                {agents[j].agent_id: agents[j] for j in agent_relationships[i]},
            ))
        for res in results:
            res.result()

        env.set_attr(attr="all_agents", value={agent.agent_id: agent for agent in agents}).result()

        self.agents = agents
        self.env = env

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
    simulator = Simulator()
    start_time = time.time()
    simulator.run()
    end_time = time.time()
    formatted_time = str(timedelta(seconds=end_time - start_time))
    logger.info(f"Total time: {formatted_time}")
