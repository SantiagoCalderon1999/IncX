from typing import Any
from vision_explanation_methods import DRISE_runner as dr
from incremental_explainer.models.base_model import BaseModel
from incremental_explainer.explainers.base_explainer import BaseExplainer
import numpy as np

class DRise(BaseExplainer):
    
    def __init__(self, model: BaseModel, num_mutants = 500) -> None:
        self._num_mutants = num_mutants
        self._model = model

    def create_saliency_map_from_path(self, results, image_path: str):
        number = 0
        results_drise = []
        while (len(results.bounding_boxes) != number):
            results_drise = dr.get_drise_saliency_map_from_path(nummasks=self._num_mutants, imagelocation=image_path, model=self._model, savename="anything", numclasses=95, max_figures=2, maskres=(4,4))
            number = len(results_drise)

        return [np.array(saliency_map['detection'])[0] for saliency_map in results_drise]
    
    def create_saliency_map(self, results, image: np.array):
        number = 0
        results_drise = []
        while (len(results.bounding_boxes) != number):
            results_drise = dr.get_drise_saliency_map(image = image, nummasks=self._num_mutants, model=self._model, maskres=(6,6))
            number = len(results_drise)

        return [np.array(saliency_map['detection'])[0] for saliency_map in results_drise]