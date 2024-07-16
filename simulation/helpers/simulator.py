import json
import os
import importlib
import glob


class BaseSimulator(object):
    """Base simulator object."""
    def __init__(self) -> None:
        pass

    def run(self):
        raise NotImplementedError

    @classmethod
    def restore(self):
        raise NotImplementedError
    
    def save(self):
        raise NotImplementedError