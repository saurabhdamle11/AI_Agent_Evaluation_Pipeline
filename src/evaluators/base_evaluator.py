from abc import ABC, abstractmethod

from src.data.schemas.evaluation import EvaluatorOutput


class BaseEvaluator(ABC):
    @abstractmethod
    async def evaluate(self, conversation: dict) -> EvaluatorOutput: ...
