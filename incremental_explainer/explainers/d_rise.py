from typing import Any
from vision_explanation_methods import DRISE_runner as dr

from vision_explanation_methods.explanations import common as od_common
from incremental_explainer.explainers.base_explainer import BaseExplainer

class DRise(BaseExplainer):
    
    def __init__(self, results, nummasks = 1000) -> None:
        self._nummasks = nummasks
        self._results = results

    def create_saliency_map(self, image_location, model: od_common.GeneralObjectDetectionModelWrapper):

        number = 0
        results_drise = []
        while (len(self._results[0].bounding_boxes) != number):
            results_drise = dr.get_drise_saliency_map(nummasks=self._nummasks, imagelocation=image_location, model= model, savename="anything", numclasses=95, max_figures=2, maskres=(16,16))
            number = len(results_drise)

        return results_drise