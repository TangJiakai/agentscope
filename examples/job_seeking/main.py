# -*- coding: utf-8 -*-
import sys, os
import random
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../src'))
os.chdir(sys.path[0])

import agentscope

from utils.utils import *
from seeker_agent import WLJob


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
        "--company_agent_configs_file",
        type=str,
        default="configs/company_agent_configs.json",
        help="The path of company agent configs file.",
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
    parser.add_argument(
        "--wl-n",
        type=int,
        default=2,
        help="The number of waiting list.",
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
    company_agents = setup_agents(args.company_agent_configs_file, recent_n=args.recent_n)
    seeker_num, job_num, company_num = len(seeker_agents), len(job_agents), len(company_agents)

    # Create id2agent mapping
    id2seeker, id2job, id2company = {}, {}, {}
    for agent in seeker_agents:
        id2seeker[agent.get_id()] = {"agent": agent}
    for agent in job_agents:
        id2job[agent.get_id()] = {"agent": agent}
    for agent in company_agents:
        id2company[agent.get_id()] = {"agent": agent}

    for job_agent in job_agents:
        job_agent.init_system_prompt(id2company[job_agent.job.company_id]['agent'].company)

    # Assign job pool to seeker agents
    for agent in seeker_agents:
        agent.job_pool = random.sample(range(1, job_num + 1), args.pool_size)
    
    print(f"Successfully initialized {seeker_num} seeker agents, {job_num} job agents, and {company_num} company agents.")

    # Start simulation
    # 1.1 [Seeker] Determine the number of job searches.
    print("=" * 50)
    print("1.1 [Seeker] Determine the number of job searches.")
    for agent in seeker_agents:
        agent.search_job_number_fun()
        # agent.search_job_number = random.choice([1,2])
    for agent in seeker_agents:
        print(f"{agent.name} wants to search {agent.search_job_number} jobs.")

    # 1.2 [Seeker] Search for jobs.
    print("=" * 50)
    print("1.2 [Seeker] Search for jobs.")
    for agent in seeker_agents:
        agent.search_jobs = random.sample(agent.job_pool, agent.search_job_number)
    for agent in seeker_agents:
        print(f"{agent.name} searches {[id2job[x]['agent'].name for x in agent.search_jobs]} jobs.")

    # 2. [Seeker] Apply for jobs.
    print("=" * 50)
    print("2. [Seeker] Apply for jobs.")
    for agent in seeker_agents:
        jobs = [id2job[x]['agent'].job for x in agent.search_jobs]
        companies = [id2company[job.company_id]['agent'].company for job in jobs]
        agent.apply_job_fun(list(zip(companies, jobs)))
        # agent.apply_jobs = random.sample(agent.search_jobs, random.choice(range(len(agent.search_jobs)))+1 if len(agent.search_jobs) > 0 else 0)
    for agent in seeker_agents:
        print(f"{agent.name} applies {[id2job[x]['agent'].name for x in agent.apply_jobs]} jobs.")

    # 3.1 [Job] Screen cv from job seekers.
    print("=" * 50)
    print("3.1 [Job] Screen cv from job seekers.")
    # Create apply_seekers list for every job agent
    for agent in job_agents:
        agent.apply_seekers = list()
    for seeker_id in id2seeker:
        seeker_agent = id2seeker[seeker_id]['agent']
        for job_id in seeker_agent.apply_jobs:
            job = id2job[job_id]['agent']
            job.apply_seekers.append(seeker_id)
    for agent in job_agents:
        agent.cv_screening_fun([id2seeker[x]['agent'].seeker for x in agent.apply_seekers], args.excess_cv_passed_n)
        # cv_passed_seekers = random.sample(agent.apply_seekers, random.choice(range(len(agent.apply_seekers)))+1 if len(agent.apply_seekers) > 0 else 0)
    
    for agent in job_agents:
        print(f"{agent.name} passes {[id2seeker[x]['agent'].name for x in agent.cv_passed_seekers]} seekers.")
    
    # 3.2 [Seeker] Notify the result of cv screening.
    print("=" * 50)
    print("3.2 [Seeker] Notify the result of cv screening.")
    for agent in seeker_agents:
        agent.cv_passed_jobs = list()
    for job_id in id2job:
        job_agent = id2job[job_id]['agent']
        for seeker_id in job_agent.cv_passed_seekers:
            seeker_agent = id2seeker[seeker_id]['agent']
            seeker_agent.cv_passed_jobs.append(job_id)

    for agent in seeker_agents:
        print(f"{agent.name} passes {[id2job[x]['agent'].name for x in agent.cv_passed_jobs]} jobs.")
    
    # 4. [Job & Seeker] Interview
    print("=" * 50)
    print("4. [Job & Seeker] Interview.")
    # TODO: 目前简化面试流程，后续如果细化整个面试过程，需要再添加moderator类，执行面试交互QA的过程
    for agent in job_agents:
        agent.interview_fun([id2seeker[x]['agent'].seeker for x in agent.cv_passed_seekers])

    # 5.1 [Job] Notify the result of interview.
    print("=" * 50)
    print("5.1 [Job] Decision the interview result.")
    for agent in job_agents:
        agent.make_decision_fun([id2seeker[x]['agent'].seeker for x in agent.cv_passed_seekers], args.wl_n)

    # 5.2 [Seeker] Notify the result of interview.
    print("=" * 50)
    print("5.2 [Seeker] Notify the result of interview.")
    for agent in seeker_agents:
        agent.offer_jobs, agent.wl_jobs, agent.reject_jobs = list(), list(), list()
        
    for job_id in id2job:
        job_agent = id2job[job_id]['agent']
        for seeker_id in job_agent.offer_seekers:
            seeker_agent = id2seeker[seeker_id]['agent']
            seeker_agent.offer_jobs.append(job_id)
        wl_n = len(job_agent.wl_seekers)
        for i, seeker_id in enumerate(job_agent.wl_seekers):
            seeker_agent.wl_jobs.append(WLJob(job_id, i+1, wl_n))
        for seeker_id in job_agent.reject_seekers:
            seeker_agent.reject_jobs.append(job_id)

    for agent in seeker_agents:
        print(f"{agent.name} receives {len(agent.offer_jobs)} offers, {len(agent.wl_jobs)} waiting list, and {len(agent.reject_jobs)} rejections.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
