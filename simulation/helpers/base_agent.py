from typing import Optional, Union, Sequence

from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.manager import ModelManager

from simulation.helpers.constants import *
from simulation.helpers.utils import *


class BaseAgent(AgentBase):
    """Base agent."""

    def __init__(self, 
                name: str, 
                model_config_name: str=None, 
                **kwargs) -> None:
        super().__init__(
            name=name,
            model_config_name=model_config_name,
        )
        self._profile = ""

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        state.pop("model", None)
        if hasattr(self, "memory"):
            memory_state = self.memory.__dict__.copy()
            memory_state["model"] = None
            state["memory"] = memory_state
        return state

    def __setstate__(self, state: object) -> None:
        self.__dict__.update(state)
        if hasattr(self, "memory_config"):
            self.memory = setup_memory(self.memory_config)
            self.memory.__dict__.update(state["memory"])
        if hasattr(self, "model_config_name"):
            self.model = ModelManager.get_instance().get_model_by_config_name(
                self.model_config_name
            )
            self.memory.model = self.model

    def set_attr(self, attr: str, value, **kwargs):
        setattr(self, attr, value)
        return "success"

    def get_attr(self, attr: str):
        return getattr(self, attr)

    def external_interview(self, observation, **kwargs):
        format_profile = PROFILE_BEGIN + self._profile + PROFILE_END
        format_instruction = INSTRUCTION_BEGIN + \
            "You are participating in a simple interview where you need to answer some questions." + \
            INSTRUCTION_END
        memory = self.memory.get_memory(get_assistant_msg(observation))
        format_memory = MEMORY_BEGIN + "\n- ".join([m["content"] for m in memory]) + MEMORY_END
        format_observation = "Question:" + observation + "Answer:"
        response = self.model.format(get_assistant_msg(
            format_instruction + format_profile + format_memory + format_observation))
        return response.text
    
    def post_intervention(self, intervention: str, **kwargs):
        self.observe(Msg("assistant", intervention, role="assistant"))
        return get_assistant_msg("success")
    
    def reply(self, x: Optional[Union[Msg, Sequence[Msg]]] = None) -> Msg:
        instruction = ""
        format_instruction = ""
        format_profile = PROFILE_BEGIN + self._profile + PROFILE_END
        observation = ""
        prompt_content = []
        memory_query = ""
        if x and hasattr(x, "instruction"):
            instruction = x.instruction
            memory_query += instruction
            format_instruction = INSTRUCTION_BEGIN + instruction + INSTRUCTION_END
            prompt_content.append(format_instruction)

        prompt_content.append(format_profile)

        if x and hasattr(x, "observation"):
            observation = x.observation
            memory_query += observation
            prompt_content.append(observation)

        if x and x["content"]:
            memory_query += x["content"]
            prompt_content.append(x["content"])

        memory = self.memory.get_memory(get_assistant_msg(memory_query))
        if len(memory) > 0:
            insert_index = -2 if len(prompt_content) > 1 else -1
            prompt_content.insert(insert_index, MEMORY_BEGIN + "\n- ".join([m["content"] for m in memory]) + MEMORY_END)

        prompt_msg = self.model.format(Msg(
            "user", 
            prompt_content, 
            role="user"
        ))
        
        if hasattr(x, "selection_num"):
            response = self.model(prompt_msg, extra_body={"selection_num": x.selection_num})
        else:
            response = self.model(prompt_msg)

        self._send_message(prompt_msg, response)
        add_memory_msg = Msg("user", instruction + observation + response.text, role="user")
        self.observe(add_memory_msg)
        return get_assistant_msg(response.text)