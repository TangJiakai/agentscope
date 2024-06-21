# -*- coding: utf-8 -*-
import sys, os
import random
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../src'))

import agentscope

from utils.utils import *


def parse_args() -> argparse.Namespace:
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool-size",
        type=int,
        default=3,
        help="The pool size of job for every seeker.",
    )
    return parser.parse_args()


def main(args) -> None:
    agentscope.init(
        project="Job Seeking Simulation",
        name="main",
        save_code=False,
        save_api_invoke=False,
        model_configs="configs/model_configs.json",
        use_monitor=False,
    )

    # Init agents
    seeker_agents = setup_agents("configs/seeker_agent_configs.json")
    interviewer_agents = setup_agents("configs/interviewer_agent_configs.json")
    seeker_num, interviewer_num = len(seeker_agents), len(interviewer_agents)

    # Create id2agent mapping
    id2seeker, id2interviewer = {}, {}
    for agent in seeker_agents:
        id2seeker[extract_ids(agent.name)[0]] = {"agent": agent}
    for agent in interviewer_agents:
        id2interviewer[extract_ids(agent.name)[0]] = {"agent": agent}

    # Assign job pool to seeker agents
    for agent in seeker_agents:
        agent.job_pool = random.sample(range(1, interviewer_num + 1), args.pool_size)
    
    print(f"Successfully initialized {seeker_num} seeker agents and {interviewer_num} interviewer agents.")

    # Start simulation
    # 1.1 [Seeker] Determine the number of job searches.
    print("=" * 50)
    print("1.1 [Seeker] Determine the number of job searches.")
    for seeker in seeker_agents:
        seeker.set_search_job_number()
    for seeker in seeker_agents:
        print(f"{seeker.name} wants to search {seeker.search_job_number} jobs.")

    # 1.2 [Seeker] Search for jobs.
    print("=" * 50)
    print("1.2 [Seeker] Search for jobs.")
    # for seeker in seeker_agents:

        


if __name__ == "__main__":
    args = parse_args()
    main(args)
