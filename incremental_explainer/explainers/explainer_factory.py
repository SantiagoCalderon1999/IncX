from incremental_explainer.explainers.explainer_enum import ExplainerEnum
from incremental_explainer.explainers.d_rise import DRise
from incremental_explainer.explainers.base_explainer import BaseExplainer
from incremental_explainer.models.base_model import BaseModel

class ExplainerFactory:
    
    def __init__(self,  model: BaseModel) -> None:
        self._model = model
    
    def get_explainer(self, explainer: ExplainerEnum) -> BaseExplainer:
        if explainer == ExplainerEnum.D_RISE:
            return DRise(self._model)

        raise ValueError(f"Unsupported explainer identifier: {explainer}")