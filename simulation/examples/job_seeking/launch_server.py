import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import argparse

import agentscope
from agentscope.server import RpcAgentServerLauncher

from simulation.helpers.constants import *
from simulation.helpers.utils import load_yaml

from agent import *

scene_path = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--base_port", type=int, default=12010)
    return parser.parse_args()


def setup_participant_agent_server(host: str, port: int) -> None:
    config = load_yaml(os.path.join(scene_path, CONFIG_DIR, SIMULATION_CONFIG))
    """Set up agent server"""
    agentscope.init(
        project=config["project_name"],
        name="server",
        runtime_id=str(port),
        save_code=False,
        save_api_invoke=False,
        model_configs=os.path.join(scene_path, CONFIG_DIR, config["model_configs_path"]),
        use_monitor=False,
    )
    assistant_server_launcher = RpcAgentServerLauncher(
        host=host,
        port=port,
        # pool_type="redis",
        max_pool_size=1638400000000,
        max_timeout_seconds=1000000000000,
    )
    assistant_server_launcher.launch(in_subprocess=False)
    assistant_server_launcher.wait_until_terminate()


if __name__ == "__main__":
    args = parse_args()
    setup_participant_agent_server(args.host, args.base_port)