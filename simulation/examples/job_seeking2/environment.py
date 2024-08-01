import os
from functools import partial
from jinja2 import Environment, FileSystemLoader

from agentscope.message import Msg
from agentscope.msghub import msghub
from agentscope.pipelines.functional import sequentialpipeline


scene_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_loader = FileSystemLoader(os.path.join(scene_path, "prompts"))
env = Environment(loader=file_loader)
Template = env.get_template("environment_prompts.j2").module


class Environment:
    def __init__(self) -> None:
        self.agents = []
        
    def get_jobs(self, job_ids):
        interviewer_agents = [self.agents[job_id] for job_id in job_ids]
        jobs = [interviewer_agent.job for interviewer_agent in interviewer_agents]
        return jobs
    
    def apply_job(self, seeker, job):
        interviewer_agent = self.agents[job.id]

        interviewer_agent.acquire_lock()
        try:
            response = interviewer_agent(Msg(
                name=seeker.name,
                content=str(seeker),
                role="user",
                fun="screening_cv",
                params={
                    "seeker": seeker,
                }
            ))
        finally:
            interviewer_agent.release_lock()

        return response
    
    def interview(self, seeker, job):
        MAX_INTERVIEW_ROUND = 3
        seeker_agent, interviewer_agent = self.agents[seeker.id], self.agents[job.id]
        
        HostMsg = partial(Msg, name="Moderator", role="assistant")
        participants = [seeker_agent, interviewer_agent]

        interviewer_agent.acquire_lock()
        try:
            hint = HostMsg(content=Template.interview_opening_statement(seeker, job))
            with msghub(participants, hint):
                x = seeker_agent()
                for _ in range(MAX_INTERVIEW_ROUND):
                    x = sequentialpipeline(participants, x)
                    if "exit" in x.content:
                        break
            
            interview_res = interviewer_agent(Msg(
                name="assistant",
                content=None,
                role="assistant",
                fun="make_decision",
                params={
                    "seeker": seeker,
                }
            ))
        finally:
            interviewer_agent.release_lock()
            
        return interview_res