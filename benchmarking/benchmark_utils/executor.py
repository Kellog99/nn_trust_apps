from abc import ABC, abstractmethod
from benchmarking.benchmark_utils.evaluator import Plan

class Executor(ABC):
    @abstractmethod
    def execute_plan(self, plan : Plan):
        raise NotImplementedError
        