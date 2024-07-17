import argparse
import os

import agentscope
from agentscope.server import RpcAgentServerLauncher

from simulation.helpers.constants import *
from simulation.helpers.utils import load_yaml

from agent import *


def parse_args() -> argparse.Namespace:
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str,default="localhost")
    parser.add_argument("--participant-num", type=int, default=100)
    parser.add_argument("--base-port", type=int, default=12010)
    parser.add_argument("--server-per-host", type=int, default=1)
    return parser.parse_args()


def setup_participant_agent_server(host: str, port: int) -> None:
    config = load_yaml(os.path.join(CONFIG_DIR, SIMULATION_CONFIG))
    """Set up agent server"""
    agentscope.init(
        project=config["project_name"],
        name="server",
        runtime_id=str(port),
        save_code=False,
        save_api_invoke=False,
        model_configs=config["model_configs_path"],
        use_monitor=False,
    )
    assistant_server_launcher = RpcAgentServerLauncher(
        host=host,
        port=port,
        max_pool_size=16384,
        custom_agent_classes=[SeekerAgent, JobAgent, CompanyAgent],
    )
    assistant_server_launcher.launch(in_subprocess=False)
    assistant_server_launcher.wait_until_terminate()


if __name__ == "__main__":
    args = parse_args()
    setup_participant_agent_server(args.host, args.base_port)