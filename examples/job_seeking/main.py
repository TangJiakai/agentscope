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
    for seeker_agent in seeker_agents:
        id2seeker[seeker_agent.get_id()] = {"agent": seeker_agent}
    for job_agent in job_agents:
        id2job[job_agent.get_id()] = {"agent": job_agent}
    for company_agent in company_agents:
        id2company[company_agent.get_id()] = {"agent": company_agent}

    for job_agent in job_agents:
        job_agent.init_system_prompt(id2company[job_agent.job.company_id]['agent'].company)
        job_agent.job.company = id2company[job_agent.job.company_id]['agent'].company

    # Assign job pool to seeker agents
    for seeker_agent in seeker_agents:
        seeker_agent.job_ids_pool = random.sample(range(1, job_num + 1), args.pool_size)
    
    print(f"Successfully initialized {seeker_num} seeker agents, {job_num} job agents, and {company_num} company agents.")

    # Start simulation
    # 1.1 [Seeker] Determine the number of job searches.
    print("=" * 50)
    print("1.1 [Seeker] Determine the number of job searches.")
    for seeker_agent in seeker_agents:
        # agent.search_job_number_fun()
        seeker_agent.search_job_number = random.choice([1,2])
    for seeker_agent in seeker_agents:
        print(f"{seeker_agent.name} wants to search {seeker_agent.search_job_number} jobs.")

    # 1.2 [Seeker] Search for jobs.
    print("=" * 50)
    print("1.2 [Seeker] Search for jobs.")
    for seeker_agent in seeker_agents:
        seeker_agent.search_job_ids = random.sample(seeker_agent.job_ids_pool, seeker_agent.search_job_number)
    for seeker_agent in seeker_agents:
        print(f"{seeker_agent.name} searches {[id2job[x]['agent'].name for x in seeker_agent.search_job_ids]} jobs.")

    # 2. [Seeker] Apply for jobs.
    print("=" * 50)
    print("2. [Seeker] Apply for jobs.")
    for seeker_agent in seeker_agents:
        jobs = [id2job[x]['agent'].job for x in seeker_agent.search_job_ids]
        # agent.apply_job_fun(jobs)
        seeker_agent.apply_job_ids = random.sample(seeker_agent.search_job_ids, random.choice(range(len(seeker_agent.search_job_ids)))+1 if len(seeker_agent.search_job_ids) > 0 else 0)
    for seeker_agent in seeker_agents:
        print(f"{seeker_agent.name} applies {[id2job[x]['agent'].name for x in seeker_agent.apply_job_ids]} jobs.")

    # 3.1 [Job] Screen cv from job seekers.
    print("=" * 50)
    print("3.1 [Job] Screen cv from job seekers.")
    # Create apply_seekers list for every job agent
    for job_agent in job_agents:
        job_agent.apply_seeker_ids = list()
    for seeker_id in id2seeker:
        seeker_agent = id2seeker[seeker_id]['agent']
        for job_id in seeker_agent.apply_job_ids:
            job_agent = id2job[job_id]['agent']
            job_agent.apply_seeker_ids.append(seeker_id)
    for job_agent in job_agents:
        # job_agent.cv_screening_fun([id2seeker[x]['agent'].seeker for x in job_agent.apply_seeker_ids], args.excess_cv_passed_n)
        job_agent.cv_passed_seeker_ids = random.sample(job_agent.apply_seeker_ids, random.choice(range(len(job_agent.apply_seeker_ids)))+1 if len(job_agent.apply_seeker_ids) > 0 else 0)
    
    for job_agent in job_agents:
        print(f"{job_agent.name} passes the cv screening for {[id2seeker[x]['agent'].name for x in job_agent.cv_passed_seeker_ids]} seekers.")
    
    # 3.2 [Seeker] Notify the result of cv screening.
    print("=" * 50)
    print("3.2 [Seeker] Notify the result of cv screening.")
    for seeker_agent in seeker_agents:
        seeker_agent.cv_passed_job_ids = list()
    for job_id in id2job:
        job_agent = id2job[job_id]['agent']
        for seeker_id in job_agent.cv_passed_seeker_ids:
            seeker_agent = id2seeker[seeker_id]['agent']
            seeker_agent.cv_passed_job_ids.append(job_id)

    for seeker_agent in seeker_agents:
        print(f"{seeker_agent.name} passes the cv screening for {[id2job[x]['agent'].name for x in seeker_agent.cv_passed_job_ids]} jobs.")
    
    # 4. [Job & Seeker] Interview
    print("=" * 50)
    print("4. [Job & Seeker] Interview.")
    # TODO: 目前简化面试流程，后续如果细化整个面试过程，需要再添加moderator类，执行面试交互QA的过程
    for job_agent in job_agents:
        job_agent.interview_fun([id2seeker[x]['agent'].seeker for x in job_agent.cv_passed_seeker_ids])

    # 5.1 [Job] Notify the result of interview.
    print("=" * 50)
    print("5.1 [Job] Decision the interview result.")
    for job_agent in job_agents:
        # job_agent.make_decision_fun([id2seeker[x]['agent'].seeker for x in job_agent.cv_passed_seeker_ids], args.wl_n)
        offer_hc = min(job_agent.hc, len(job_agent.cv_passed_seeker_ids))
        wl_n = min(args.wl_n, len(job_agent.cv_passed_seeker_ids) - offer_hc)
        job_agent.offer_seeker_ids = random.sample(job_agent.cv_passed_seeker_ids, offer_hc)
        job_agent.wl_seeker_ids = random.sample(list(set(job_agent.cv_passed_seeker_ids) - set(job_agent.offer_seeker_ids)), wl_n)
        job_agent.reject_seeker_ids = list(set(job_agent.cv_passed_seeker_ids) - set(job_agent.offer_seeker_ids) - set(job_agent.wl_seeker_ids))

    for job_agent in job_agents:
        print(f"{job_agent.name} offers {[id2seeker[x]['agent'].name for x in job_agent.offer_seeker_ids]}, waitlists {[id2seeker[x]['agent'].name for x in job_agent.wl_seeker_ids]}, and rejects {[id2seeker[x]['agent'].name for x in job_agent.reject_seeker_ids]}.")

    # 5.2 [Seeker] Notify the result of interview.
    print("=" * 50)
    print("5.2 [Seeker] Notify the result of interview.")
    for seeker_agent in seeker_agents:
        seeker_agent.offer_job_ids, seeker_agent.wl_jobs, seeker_agent.fail_job_ids = list(), list(), list()
        
    for job_id in id2job:
        job_agent = id2job[job_id]['agent']
        for seeker_id in job_agent.offer_seeker_ids:
            seeker_agent = id2seeker[seeker_id]['agent']
            seeker_agent.offer_job_ids.append(job_id)
        wl_n = len(job_agent.wl_seeker_ids)
        for i, seeker_id in enumerate(job_agent.wl_seeker_ids):
            seeker_agent = id2seeker[seeker_id]['agent']
            seeker_agent.wl_jobs.append(WLJob(job_id, i+1, wl_n))
        for seeker_id in job_agent.reject_seeker_ids:
            seeker_agent = id2seeker[seeker_id]['agent']
            seeker_agent.fail_job_ids.append(job_id)

    for seeker_agent in seeker_agents:
        print(f"{seeker_agent.name} receives {len(seeker_agent.offer_job_ids)} offers, {len(seeker_agent.wl_jobs)} waiting list, and {len(seeker_agent.fail_job_ids)} failed jobs.")

    # 6.1 [Seeker] Make decision
    print("=" * 50)
    print("6.1 [Seeker] Make decision.")
    for seeker_agent in seeker_agents:
        seeker_agent.make_decision_fun(id2job)

    for seeker_agent in seeker_agents:
        if seeker_agent.decision == 0:  # No any offers, and continue to search for jobs
            print(f"{seeker_agent.name} has no any offers, and continues to search for jobs.")
        elif seeker_agent.decision == 1:    # Accept the offer
            print(f"{seeker_agent.name} accepts the offer {id2job[seeker_agent.final_offer_id]['agent'].name}.")
        elif seeker_agent.decision == 2:    # Wait for the waitlist offer
            print(f"{seeker_agent.name} rejects all offers, and waits for {[id2job[x]['agent'].name for x in seeker_agent.wl_job_ids]}.")
        elif seeker_agent.decision == 3:    # Reject all offers and waiting list, and continue to search for jobs
            print(f"{seeker_agent.name} rejects all offers and waiting list, and continues to search for jobs.")
    
    # 6.2 [Job] Complete the handshake agreements or adjust the waitlist accordingly.
    print("=" * 50)
    print("6.2 [Job] Complete the handshake agreements and adjust the waitlist accordingly.")
    


if __name__ == "__main__":
    args = parse_args()
    main(args)
