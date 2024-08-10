from incx.dependencies.d_rise.vision_explanation_methods import (
    DRISE_runner as dr,
)
from incx.models.base_model import BaseModel
from incx.explainers.base_explainer import BaseExplainer
import numpy as np
import torchvision.transforms as transforms


class DRise(BaseExplainer):
    def __init__(self, model: BaseModel, num_mutants=1000) -> None:
        self._num_mutants = num_mutants
        self._model = model

    def create_saliency_map_from_path(self, results, image_path: str):
        number = 0
        results_drise = []
        while len(results.bounding_boxes) != number:
            results_drise = dr.get_drise_saliency_map_from_path(
                nummasks=self._num_mutants,
                imagelocation=image_path,
                model=self._model,
                savename="anything",
                numclasses=95,
                max_figures=2,
                maskres=(4, 4),
            )
            number = len(results_drise)

        return [
            np.array(saliency_map["detection"])[0] for saliency_map in results_drise
        ]

    def create_saliency_map(self, image: np.array):
        number = 0
        results_drise = []
        transform = transforms.Compose([transforms.ToTensor()])
        results = self._model.predict(transform(image).unsqueeze(0))[0]
        counter = 0
        while len(results.bounding_boxes) != number:
            results_drise = dr.get_drise_saliency_map(
                image=image,
                nummasks=self._num_mutants,
                model=self._model,
                maskres=(6, 6),
                seed_start=counter*self._num_mutants,
            )
            number = len(results_drise)
            counter += 1

        return [
            np.array(saliency_map["detection"])[0] for saliency_map in results_drise
        ]
