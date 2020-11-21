from typing import Dict
from abc import ABC, abstractmethod

# stateless experiment base -- a guide which methods to
class BaseTool(ABC):
    
    @abstractmethod
    def configure_argument_parser(self, parser):
        pass

    @abstractmethod
    def run(self, args):
        pass
    
    def get_help(self) -> str:
        return "Override BaseExperiment.get_help to change this help!"