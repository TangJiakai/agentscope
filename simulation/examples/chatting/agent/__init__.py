import os
import importlib
from .chatting_agent import ChatRoomAgentStates

module_names = [
    f[:-3] for f in os.listdir(os.path.dirname(__file__)) if f.endswith("_agent.py")
]

ALL_AGENT_STATES = {
    "ChatRoomAgent": ChatRoomAgentStates,
}

__all__ = ["ALL_AGENT_STATES"]

for module_name in module_names:
    module = importlib.import_module(f".{module_name}", package=__name__)
    globals().update(
        {
            name: getattr(module, name)
            for name in dir(module)
            if not name.startswith("_")
        }
    )
    __all__.extend([name for name in dir(module) if not name.startswith("_")])
