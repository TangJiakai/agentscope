import random
import os
import requests
import jinja2
from loguru import logger

from agentscope.agents import RpcAgent
from agentscope.message import Msg
from agentscope.rpc import async_func

from simulation.examples.recommendation.environment.env import RecommendationEnv
from simulation.helpers.base_agent import BaseAgent
from simulation.helpers.utils import *
from simulation.helpers.constants import *


scene_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_loader = jinja2.FileSystemLoader(os.path.join(scene_path, "prompts"))
env = jinja2.Environment(loader=file_loader)
Template = env.get_template("recuser_prompts.j2").module


RecUserAgentStates = [
    "idle",
    "watching",
    "chatting",
    "posting",
]

def set_state(flag: str):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            init_state = self.state
            self.state = flag
            try:
                return func(self, *args, **kwargs)
            finally:
                self.state = init_state

        return wrapper
    return decorator


class RecUser(object):
    def __init__(self, 
                name: str, 
                gender: str, 
                age: int,
                traits: str,
                status: str,
                interest: str,
                feature: str,
        ):
        self.name = name
        self.gender = gender
        self.age = age
        self.traits = traits
        self.status = status
        self.interest = interest
        self.feature = feature

    def __str__(self):
        return (
            f"Name: {self.name}\n"
            f"Gender: {self.gender}\n"
            f"Age: {self.age}\n"
            f"Traits: {self.traits}\n"
            f"Status: {self.status}\n"
            f"Interest: {self.interest}\n"
            f"Feature: {self.feature}\n"
        )


class RecUserAgent(BaseAgent):
    """recuser agent."""

    def __init__(
        self,
        name: str,
        model_config_name: str,
        memory_config: dict,
        embedding_api: str,
        gender: str, 
        age: int,
        traits: str,
        status: str,
        interest: str,
        feature: str,
        env: RecommendationEnv,
        relationship: dict[str, RpcAgent] = {},
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            model_config_name=model_config_name,
        )
        self.memory_config = memory_config
        self.embedding_api = embedding_api
        self.memory = setup_memory(memory_config)
        self.memory.embedding_api = embedding_api
        self.memory.model = self.model
        self.env = env
        self.relationship = relationship

        self.recuser = RecUser(name, gender, age, traits, status, interest, feature)
        self._update_profile()
        self._state = "idle"

    def _update_profile(self):
        self._profile = self.recuser.__str__()

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_value):
        if hasattr(self, "backend_server_url"):
            if new_value not in RecUserAgentStates:
                raise ValueError(f"Invalid state: {new_value}")
            self._state = new_value
            url = f"{self.backend_server_url}/api/state"
            resp = requests.post(
                url, json={"agent_id": self.agent_id, "state": new_value}
            )
            if resp.status_code != 200:
                logger.error(f"Failed to set state: {self.agent_id} -- {new_value}")

    def generate_feeling(self, movie):
        instruction = Template.generate_feeling_instruction()
        observation = Template.generate_feeling_observation(movie)
        msg = get_assistant_msg()
        msg.instruction = instruction
        msg.observation = observation
        feeling = self(msg)["content"]
        
        logger.info(f"[{self.name}] feels {feeling}")
        return feeling

    def rating_item(self, movie):
        instruction = Template.rating_item_instruction()
        guided_choice = [
            "Rating 1: Very poor quality, unenjoyable, with major flaws.",
            "Rating 2: Noticeable issues, disappointing, with a few redeeming moments.",
            "Rating 3: Decent but unremarkable, watchable with some strengths and weaknesses.",
            "Rating 4: Well-made and enjoyable, with minor flaws.",
            "Rating 5: Outstanding in all aspects, highly enjoyable, and memorable."
        ]
        observation = Template.rating_item_observation(movie, guided_choice)
        msg = get_assistant_msg()
        msg.instruction = instruction
        msg.observation = observation
        msg.guided_choice = guided_choice
        action = self(msg)["content"]

        logger.info(f"[{self.name}] rated {action}")

        return action

    @set_state("watching")
    def recommend(self):
        user_info = self._profile + \
            "\nMemory:" + "\n- ".join([m["content"] for m in self.memory.get_memory()])
        guided_choice = self.env.recommend4user(user_info)
        instruction = Template.recommend_instruction()
        observation = Template.make_choice_observation(guided_choice)
        msg = get_assistant_msg()
        msg.instruction = instruction
        msg.observation = observation
        msg.guided_choice = guided_choice
        action = self(msg)["content"].split(":")[0]

        logger.info(f"[{self.name}] selected {action}")

        feeling = self.generate_feeling(action)
        rating = self.rating_item(action)

    @set_state("chatting")
    def conversation(self):
        friend_agent_id = random.choice(list(self.relationship.keys()))
        friend_agent = self.relationship[friend_agent_id]
        announcement = Template.conversation_instruction()
        dialog_observation = self.chat(announcement, [self, friend_agent])

        self.observe(get_assistant_msg(announcement + dialog_observation))
        friend_agent.observe(get_assistant_msg(announcement + dialog_observation))

        logger.info(f"[{self.name}] had a conversation with {friend_agent_id}: {dialog_observation}")

        return dialog_observation
    
    @set_state("chatting")
    def respond_conversation(self, observation: str):
        instruction = Template.conversation_instruction()
        format_instruction = INSTRUCTION_BEGIN + instruction + INSTRUCTION_END
        format_profile = PROFILE_BEGIN + self._profile + PROFILE_END
        memory = self.memory.get_memory(get_assistant_msg(instruction))
        memory_content = get_memory_until_limit(memory, "\n".join(memory))
        format_memory = MEMORY_BEGIN + memory_content + MEMORY_END
        response = self.model(self.model.format(Msg(
            "user",
            format_instruction + format_profile + format_memory + observation + f"\n{self.name}:",
            role="user",
        )))
        return get_assistant_msg(f"\n{self.name}: {response.text}")
    
    @set_state("posting")
    def post(self):
        instruction = Template.post_instruction()
        msg = get_assistant_msg()
        msg.instruction = instruction
        msg.observation = "Please give your post content."
        response = self(msg)["content"]
        
        for agent in self.relationship.values():
            agent.observe(get_assistant_msg(f"{self.name} posted: {response}"))

        logger.info(f"[{self.name}] posted: {response}")

        return response

    @async_func
    def run(self, **kwargs):
        instruction = Template.start_action_instruction()
        guided_choice = [
            "Recommend: Request the website to recommend a batch of movies to watch.",
            "Conversation: Start a conversation with a good friend about a movie you've recently heard about or watched.",
            "Post: Post in your social circle expressing your recent thoughts on movie-related topics."
        ]
        observation = Template.make_choice_observation(guided_choice)
        msg = get_assistant_msg(instruction + observation)
        msg.instruction = instruction
        msg.observation = observation
        msg.guided_choice = guided_choice
        
        action = self(msg)["content"].split(":")[0].strip().lower()
        getattr(self, action)()
        
        return "success"