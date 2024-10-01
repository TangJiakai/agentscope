import os
import dill
from tqdm import tqdm
from loguru import logger

import agentscope
from agentscope.manager import FileManager

from simulation.helpers.constants import *
from agentscope.constants import _DEFAULT_SAVE_DIR
from simulation.examples.recommendation.agent import *
from simulation.helpers.emb_service import *
from simulation.helpers.utils import *

CUR_ROUND = 1


class BaseSimulator:
    def __init__(self, scene_path):
        super().__init__()
        self.scene_path = scene_path
        self.config = load_yaml(os.path.join(scene_path, CONFIG_DIR, SIMULATION_CONFIG))
        self.cur_round = 1
        self.resume = False
        self.agent_save_state = None
        self._from_scratch()

    def _from_scratch(self):
        self._init_agentscope()

        if self.config["load_simulator_path"] is not None:
            logger.info(f"Load simulator from {self.config['load_simulator_path']}")
            loaded_simulator = self.load(self.config["load_simulator_path"])
            self.__dict__.update(loaded_simulator.__dict__)
            self.resume = True
            self._init_agents_envs(resume=True)
            logger.info("Load simulator successfully")
        else:
            self._init_agents_envs()

        save_configs(self.config)

    def _init_agentscope(self):
        agentscope.init(
            project=self.config["project_name"],
            save_code=False,
            save_api_invoke=False,
            model_configs=os.path.join(self.scene_path, CONFIG_DIR, MODEL_CONFIG),
            use_monitor=False,
            save_dir=os.path.join(_DEFAULT_SAVE_DIR, self.config["project_name"]),
            runtime_id=self.config["runtime_id"],
        )

    def _set_env4agents(self):
        logger.info("Set all_agents for envs")
        agent_dict = {agent.agent_id: agent for agent in self.agents}
        results = []
        for env in self.envs:
            results.append(env.set_attr(attr="all_agents", value=agent_dict))
        for res in tqdm(results, total=len(self.envs), desc="Set all_agents for envs"):
            res.result()

    def _init_agents_envs(self, resume=False):
        raise NotImplementedError

    def _one_round(self):
        results = []
        for agent in self.agents:
            results.append(agent.run())
        outputs = []
        for res in results:
            output = res.result()
            outputs.append(output)
            logger.info(output)
        return outputs

    def run(self):
        raise NotImplementedError

    def load(file_path):
        with open(file_path, "rb") as f:
            return dill.load(f)

    def get_save_state(self):
        raise NotImplementedError

    def save(self):
        try:
            file_manager = FileManager.get_instance()
            save_path = os.path.join(
                file_manager.run_dir, f"ROUND-{self.cur_round}.pkl"
            )
            self.agent_save_state, self.env_save_state = self.get_save_state()
            self.cur_round += 1
            with open(save_path, "wb") as f:
                dill.dump(self, f)
            logger.info(f"Saved simulator to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save simulator: {e}")
