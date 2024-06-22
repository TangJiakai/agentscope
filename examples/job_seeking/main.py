# -*- coding: utf-8 -*-
import sys, os
import random
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../src'))
os.chdir(sys.path[0])

import agentscope

from utils.utils import *


def parse_args() -> argparse.Namespace:
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-configs",
        type=str,
        default="configs/model_configs.json",
        help="The path of model configs file.",
    )
    parser.add_argument(
        "--seeker_agent_configs_file",
        type=str,
        default="configs/seeker_agent_configs.json",
        help="The path of seeker agent configs file.",
    )
    parser.add_argument(
        "--job_agent_configs_file",
        type=str,
        default="configs/job_agent_configs.json",
        help="The path of job agent configs file.",
    )

    parser.add_argument(
        "--pool-size",
        type=int,
        default=3,
        help="The pool size of job for every seeker.",
    )
    parser.add_argument(
        "--recent-n",
        type=int,
        default=5,
        help="The number of recent memory for every prompt.",
    )
    
    parser.add_argument(
        "--excess-cv-passed-n",
        type=int,
        default=2,
        help="The number of excess cv passed.",
    )
    return parser.parse_args()


def main(args) -> None:
    agentscope.init(
        project="Job Seeking Simulation",
        name="main",
        save_code=False,
        save_api_invoke=False,
        model_configs=args.model_configs,
        use_monitor=False,
    )

    # Init agents
    seeker_agents = setup_agents(args.seeker_agent_configs_file, recent_n=args.recent_n)
    job_agents = setup_agents(args.job_agent_configs_file, recent_n=args.recent_n)
    seeker_num, job_num = len(seeker_agents), len(job_agents)

    # Create id2agent mapping
    id2seeker, id2job = {}, {}
    for agent in seeker_agents:
        id2seeker[extract_ids(agent.name)[0]] = {"agent": agent}
    for agent in job_agents:
        id2job[extract_ids(agent.name)[0]] = {"agent": agent}

    # Assign job pool to seeker agents
    for agent in seeker_agents:
        agent.job_pool = random.sample(range(1, job_num + 1), args.pool_size)
    
    print(f"Successfully initialized {seeker_num} seeker agents and {job_num} job agents.")

    # Start simulation
    # 1.1 [Seeker] Determine the number of job searches.
    print("=" * 50)
    print("1.1 [Seeker] Determine the number of job searches.")
    for seeker in seeker_agents:
        seeker.search_job_number_fun()
    for seeker in seeker_agents:
        print(f"{seeker.name} wants to search {seeker.search_job_number} jobs.")

    # 1.2 [Seeker] Search for jobs.
    print("=" * 50)
    print("1.2 [Seeker] Search for jobs.")
    for seeker in seeker_agents:
        seeker.search_jobs = random.sample(seeker.job_pool, seeker.search_job_number)
    for seeker in seeker_agents:
        print(f"{seeker.name} searches {[id2job[x]['agent'].name for x in seeker.search_jobs]} jobs.")

    # 2. [Seeker] Apply for jobs.
    print("=" * 50)
    print("2. [Seeker] Apply for jobs.")
    for seeker in seeker_agents:
        seeker.apply_job_fun([id2job[x]['agent'] for x in seeker.search_jobs])
    for seeker in seeker_agents:
        print(f"{seeker.name} applies {[id2job[x]['agent'].name for x in seeker.apply_jobs]} jobs.")

    # 3. [Job] Notify the cv screening results.
    print("=" * 50)
    print("3. [Job] Notify the cv screening results.")
    # Create apply_seekers list for every job agent
    for job in job_agents:
        job.apply_seekers = list()
    for seeker_id in id2seeker:
        seeker_agent = id2seeker[seeker_id]['agent']
        for job_id in seeker_agent.apply_jobs:
            job = id2job[job_id]['agent']
            job.apply_seekers.append(seeker_id)
    for job in job_agents:
        job.cv_screening_fun([id2seeker[x]['agent'] for x in job.apply_seekers], args.excess_cv_passed_n)


if __name__ == "__main__":
    args = parse_args()
    main(args)
