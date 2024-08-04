from abc import ABC, abstractmethod
from incx.models.base_model import BaseModel


class BaseExplainer(ABC):
    def __init__(self, model: BaseModel):
        self._model = model

    @abstractmethod
    def create_saliency_map(self, image):
        pass
