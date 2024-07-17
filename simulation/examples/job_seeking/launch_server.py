import argparse

import agentscope
from agentscope.server import RpcAgentServerLauncher

from agent import *

def parse_args() -> argparse.Namespace:
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hosts",
        type=str,
        nargs="+",
        default=["localhost"],
    )
    parser.add_argument("--participant-num", type=int, default=100)
    parser.add_argument("--base-port", type=int, default=12010)
    parser.add_argument(
        "--server-per-host",
        type=int,
    )
    return parser.parse_args()



def setup_participant_agent_server(host: str, port: int) -> None:
    """Set up agent server"""
    agentscope.init(
        project="simulation",
        name="server",
        runtime_id=str(port),
        save_code=False,
        save_api_invoke=False,
        model_configs="configs/model_configs.json",
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