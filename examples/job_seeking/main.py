# -*- coding: utf-8 -*-
import sys, os
import random
import argparse
from copy import deepcopy
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
        "--company_agent_configs_file",
        type=str,
        default="configs/company_agent_configs.json",
        help="The path of company agent configs file.",
    )

    parser.add_argument(
        "--turn-n",
        type=int,
        default=2,
        help="The max number of turns.",
    )
    parser.add_argument(    # avoid infinite loop
        "--make-decision-turn-n",
        type=int,
        default=2,
        help="The max number of make decision turns.",
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


def single_turn_make_decision_fun(seeker_agents, job_agents, id2seeker, id2job):
    # 6.1 [Seeker] Make decision
    print("6.1 [Seeker] Make decision.")
    for seeker_agent in seeker_agents:
        seeker_agent.make_decision_fun(id2job)
        # if len(seeker_agent.offer_job_ids) == 0 and len(seeker_agent.wl_jobs_dict) == 0:
        #     seeker_agent.decision = 0
        #     seeker_agent.final_offer_id = None
        #     seeker_agent.reject_offer_job_ids, seeker_agent.reject_wl_job_ids = list(), list()
        # else:
        #     if len(seeker_agent.offer_job_ids) > 0:
        #         if len(seeker_agent.wl_jobs_dict) > 0:
        #             decision = random.choice([1, 2, 3])
        #         else:
        #             decision = random.choice([1, 3])
        #     else:
        #         decision = 2
        #     if decision == 1:
        #         final_offer_id = random.choice(seeker_agent.offer_job_ids)
        #         seeker_agent.decision = 1
        #         seeker_agent.final_offer_id = final_offer_id
        #         seeker_agent.reject_offer_job_ids = list(set(seeker_agent.offer_job_ids) - set([final_offer_id]))
        #         seeker_agent.reject_wl_job_ids = [x for x in seeker_agent.wl_jobs_dict]
        #     elif decision == 2:
        #         seeker_agent.decision = 2
        #         seeker_agent.final_offer_id = None
        #         seeker_agent.offer_job_ids = list()
        #         seeker_agent.reject_offer_job_ids = seeker_agent.offer_job_ids
        #         seeker_agent.reject_wl_job_ids = list()
        #     else:
        #         seeker_agent.decision = 3
        #         seeker_agent.final_offer_id = None
        #         seeker_agent.offer_job_ids = list()
        #         seeker_agent.wl_jobs_dict = dict()
        #         seeker_agent.reject_offer_job_ids = seeker_agent.offer_job_ids
        #         seeker_agent.reject_wl_job_ids = [x for x in seeker_agent.wl_jobs_dict]

    for seeker_agent in seeker_agents:
        if seeker_agent.decision == 0:  # No any offers, and continue to search for jobs
            seeker_agent.memory_info["final_decision"] = 3
            print(f"{seeker_agent.name} has no any offers, and continues to search for jobs.")
        elif seeker_agent.decision == 1:    # Accept the offer
            seeker_agent.memory_info["final_decision"] = 1 if seeker_agent.memory_info["waiting_time"] == 0 else 2
            seeker_agent.memory_info["final_offer"] = id2job[seeker_agent.final_offer_id]['agent'].job
            print(f"{seeker_agent.name} accepts the offer {id2job[seeker_agent.final_offer_id]['agent'].name}.")
        elif seeker_agent.decision == 2:    # Wait for the waitlist offer
            seeker_agent.memory_info["waiting_time"] += 1
            print(f"{seeker_agent.name} rejects all offers, and waits for {[id2job[x]['agent'].name for x in seeker_agent.wl_jobs_dict]}.")
        elif seeker_agent.decision == 3:    # Reject all offers and waiting list, and continue to search for jobs
            seeker_agent.memory_info["final_decision"] = 4 if seeker_agent.memory_info["waiting_time"] == 0 else 5
            print(f"{seeker_agent.name} rejects all offers and waiting list, and continues to search for jobs.")
    
    # 6.2 [Job] Complete the handshake agreements or adjust the waitlist accordingly.
    print("=" * 50)
    print("6.2 [Job] Complete the handshake agreements and adjust the waitlist accordingly.")
    for seeker_agent in seeker_agents:
        seeker_id = seeker_agent.get_id()
        if seeker_agent.decision == 1: # Accept the offer
            job_agent = id2job[seeker_agent.final_offer_id]['agent']
            job_agent.hc -= 1
            print(job_agent.offer_seeker_ids)
            print(seeker_id)
            job_agent.offer_seeker_ids.remove(seeker_id)
            job_agent.memory_info["final_offer_seeker"].append(id2seeker[seeker_id]['agent'].seeker)
        for job_id in seeker_agent.reject_offer_job_ids:
            job_agent = id2job[job_id]['agent']
            job_agent.offer_seeker_ids.remove(seeker_id)
            if len(job_agent.wl_seeker_ids) > 0:
                wl_seeker_id = job_agent.wl_seeker_ids.pop(0)
                job_agent.offer_seeker_ids.append(wl_seeker_id)
        for job_id in seeker_agent.reject_wl_job_ids:
            job_agent = id2job[job_id]['agent']
            job_agent.wl_seeker_ids.remove(seeker_id)
            if len(job_agent.wl_seeker_ids) > 0:
                wl_seeker_id = job_agent.wl_seeker_ids.pop(0)
                job_agent.offer_seeker_ids.append(wl_seeker_id)

    for seeker_agent in seeker_agents:
        seeker_agent.offer_job_ids, seeker_agent.wl_jobs_dict = list(), dict()
    
    for job_id in id2job:
        job_agent = id2job[job_id]['agent']
        for seeker_id in job_agent.offer_seeker_ids:
            seeker_agent = id2seeker[seeker_id]['agent']
            seeker_agent.offer_job_ids.append(job_id)
        wl_n = len(job_agent.wl_seeker_ids)
        for i, seeker_id in enumerate(job_agent.wl_seeker_ids):
            seeker_agent = id2seeker[seeker_id]['agent']
            seeker_agent.wl_jobs_dict[job_id] = {"rank": i+1, "wl_n": wl_n}
        
    for job_agent in job_agents:
        print(f"{job_agent.name} offers {[id2seeker[x]['agent'].name for x in job_agent.offer_seeker_ids]}, waitlists {[id2seeker[x]['agent'].name for x in job_agent.wl_seeker_ids]}.")
        

def single_turn(args, all_seeker_agents, job_agents, company_agents, id2seeker, id2job, id2company):

    # TODO: 求职者状态转换，如果不准备找工作，此轮可以无视此人
    seeker_agents=[]
    for seeker_agent in all_seeker_agents:
        seeker_agent.determine_status()
        if seeker_agent.finding:
            seeker_agents.append(seeker_agent)
    seeker_num,all_seeker_num, job_num, company_num =len(seeker_agents), len(all_seeker_agents), len(job_agents), len(company_agents)
    # TODO: 需要找工作的求职者的 job_ids_pool 使用Faiss进行相似度搜索，找到若干工作作为当前大轮的初始职位池
    # Assign job pool to seeker agents
    for seeker_agent in seeker_agents:
        seeker_agent.job_ids_pool = random.sample(range(1, job_num + 1), args.pool_size)
    
    print(f"Successfully initialized a total of {all_seeker_num} seeker agents, {seeker_num} is finding job, {job_num} job agents, and {company_num} company agents.")

    # Start simulation
    # 1.1 [Seeker] Determine the number of job searches.
    print("=" * 50)
    print("1.1 [Seeker] Determine the number of job searches.")
    for seeker_agent in seeker_agents:
        seeker_agent.search_job_number_fun()
        # seeker_agent.search_job_number = random.choice([1,2])
    for seeker_agent in seeker_agents:
        seeker_agent.memory_info["search_job_number"] = seeker_agent.search_job_number
        print(f"{seeker_agent.name} wants to search {seeker_agent.search_job_number} jobs.")

    # 1.2 [Seeker] Search for jobs.
    print("=" * 50)
    print("1.2 [Seeker] Search for jobs.")
    for seeker_agent in seeker_agents:
        seeker_agent.search_job_ids = random.sample(seeker_agent.job_ids_pool, seeker_agent.search_job_number)
    for seeker_agent in seeker_agents:
        seeker_agent.memory_info["search_jobs"] = [id2job[x]['agent'].job for x in seeker_agent.search_job_ids]
        print(f"{seeker_agent.name} searches {[id2job[x]['agent'].name for x in seeker_agent.search_job_ids]} jobs.")

    # 2. [Seeker] Apply for jobs.
    print("=" * 50)
    print("2. [Seeker] Apply for jobs.")
    for seeker_agent in seeker_agents:
        jobs = [id2job[x]['agent'].job for x in seeker_agent.search_job_ids]
        seeker_agent.apply_job_fun(jobs)
        # seeker_agent.apply_job_ids = random.sample(seeker_agent.search_job_ids, random.choice(range(len(seeker_agent.search_job_ids)))+1 if len(seeker_agent.search_job_ids) > 0 else 0)
    for seeker_agent in seeker_agents:
        seeker_agent.memory_info["apply_job_ids"] = seeker_agent.apply_job_ids
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
        job_agent.cv_screening_fun([id2seeker[x]['agent'].seeker for x in job_agent.apply_seeker_ids], args.excess_cv_passed_n)
        job_agent.memory_info["apply_seekers"] = [id2seeker[x]['agent'].seeker for x in job_agent.apply_seeker_ids]
        job_agent.memory_info["cv_passed_seeker_ids"] = job_agent.cv_passed_seeker_ids
        # job_agent.cv_passed_seeker_ids = random.sample(job_agent.apply_seeker_ids, random.choice(range(len(job_agent.apply_seeker_ids)))+1 if len(job_agent.apply_seeker_ids) > 0 else 0)
    
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
        seeker_agent.memory_info["cv_passed_job_ids"] = seeker_agent.cv_passed_job_ids
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
        job_agent.make_decision_fun([id2seeker[x]['agent'].seeker for x in job_agent.cv_passed_seeker_ids], args.wl_n)
        # offer_hc = min(job_agent.hc, len(job_agent.cv_passed_seeker_ids))
        # wl_n = min(args.wl_n, len(job_agent.cv_passed_seeker_ids) - offer_hc)
        # job_agent.offer_seeker_ids = random.sample(job_agent.cv_passed_seeker_ids, offer_hc)
        # job_agent.wl_seeker_ids = random.sample(list(set(job_agent.cv_passed_seeker_ids) - set(job_agent.offer_seeker_ids)), wl_n)
        # job_agent.reject_seeker_ids = list(set(job_agent.cv_passed_seeker_ids) - set(job_agent.offer_seeker_ids) - set(job_agent.wl_seeker_ids))

    for job_agent in job_agents:
        job_agent.memory_info["offer_seeker_ids"] = deepcopy(job_agent.offer_seeker_ids)
        job_agent.memory_info["wl_seeker_ids"] = deepcopy(job_agent.wl_seeker_ids)
        print(f"{job_agent.name} offers {[id2seeker[x]['agent'].name for x in job_agent.offer_seeker_ids]}, waitlists {[id2seeker[x]['agent'].name for x in job_agent.wl_seeker_ids]}, and rejects {[id2seeker[x]['agent'].name for x in job_agent.reject_seeker_ids]}.")

    # 5.2 [Seeker] Notify the result of interview.
    print("=" * 50)
    print("5.2 [Seeker] Notify the result of interview.")
    for seeker_agent in seeker_agents:
        seeker_agent.offer_job_ids, seeker_agent.wl_jobs_dict, seeker_agent.fail_job_ids = list(), dict(), list()
        
    for job_id in id2job:
        job_agent = id2job[job_id]['agent']
        for seeker_id in job_agent.offer_seeker_ids:
            seeker_agent = id2seeker[seeker_id]['agent']
            seeker_agent.offer_job_ids.append(job_id)
        wl_n = len(job_agent.wl_seeker_ids)
        for i, seeker_id in enumerate(job_agent.wl_seeker_ids):
            seeker_agent = id2seeker[seeker_id]['agent']
            seeker_agent.wl_jobs_dict[job_id] = {"rank": i+1, "wl_n": wl_n}
        for seeker_id in job_agent.reject_seeker_ids:
            seeker_agent = id2seeker[seeker_id]['agent']
            seeker_agent.fail_job_ids.append(job_id)

    for seeker_agent in seeker_agents:
        seeker_agent.memory_info["initial_offer_job_ids"] = deepcopy(seeker_agent.offer_job_ids)
        seeker_agent.memory_info["initial_wl_jobs_dict"] = deepcopy(seeker_agent.wl_jobs_dict)
        print(f"{seeker_agent.name} receives {len(seeker_agent.offer_job_ids)} offers, {len(seeker_agent.wl_jobs_dict)} waiting list, and {len(seeker_agent.fail_job_ids)} failed jobs.")
    
    # 6. [Seeker & Job] Make decision
    print("=" * 50)
    cur_seeker_agents = seeker_agents
    for i in range(args.make_decision_turn_n):
        print(f"Make decision turn {i+1}")

        single_turn_make_decision_fun(cur_seeker_agents, job_agents, id2seeker, id2job)
        cur_seeker_agents = [x for x in cur_seeker_agents if x.decision == 2]

        # Check if exists seeker agents that have offers
        stop_flag = True
        for seeker_agent in cur_seeker_agents:
            if len(seeker_agent.offer_job_ids) > 0:
                stop_flag = False
                break
        if stop_flag:
            break

    # [Seeker & Job] Add memory
    print("=" * 50)
    print("[Seeker & Job] Add memory.")
    for agent in seeker_agents + job_agents:
        agent.add_memory()

    # 7. [Seeker & Job] The seekers and jobs refresh the information.
    print("=" * 50)
    # 7.1 [Seeker] Refresh information
    print("7.1 [Seeker] Refresh information.")
    # TODO: 简历更新等
    for seeker_agent in seeker_agents:
        seeker_agent.update_fun()

    # 7.2 [Company] Refresh hc
    print("=" * 50)
    # TODO: 企业动态发布职位，更新hc等，需要注意如果岗位新增或者减少，那job相关变量都需要重新定义，id2job需要重新映射
    print("7.2 [Company] Refresh information.")
    for company_agent in company_agents:
        company_agent.update_fun()

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

    for i in range(args.turn_n):
        print(f"Turn {i+1}")
        single_turn(args, seeker_agents, job_agents, company_agents, id2seeker, id2job, id2company)

if __name__ == "__main__":
    args = parse_args()
    main(args)
