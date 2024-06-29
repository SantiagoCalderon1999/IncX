from incremental_explainer.models.base_model import BaseModel
from incremental_explainer.explainers.base_explainer import BaseExplainer
from incremental_explainer.tracking.so_tracker import SoTracker
from incremental_explainer.explainers.explainer_factory import ExplainerFactory
from incremental_explainer.explainers.explainer_enum import ExplainerEnum
from incremental_explainer.models.model_factory import ModelFactory
from incremental_explainer.models.model_enum import ModelEnum
import cv2
from torchvision import transforms

class IncRex:
    
    def __init__(self, model: BaseModel, explainer: BaseExplainer, object_index) -> None:
        self._prev_saliency_map = []
        self._previous_bounding_box = []
        self._frame_number = 0
        self._model = model
        self._explainer = explainer
        self._explanation_tracker = None
        self._object_index = object_index
    
    def explain(self, image):
        
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_t = transform(image)

        prediction = self._model.predict([img_t])
        prediction = prediction[0]
        
        if self._frame_number == 0:
            saliency_map = self._explainer.create_saliency_map(prediction, image)[self._object_index]
            self._explanation_tracker = SoTracker(saliency_map, prediction, self._object_index)
            bounding_box = prediction.bounding_boxes[self._object_index]
        else:
            saliency_map, bounding_box = self._explanation_tracker.compute_tracked_explanation(image, prediction)
        
        self._frame_number += 1
        
        self._prev_saliency_map = saliency_map
        return saliency_map, bounding_box