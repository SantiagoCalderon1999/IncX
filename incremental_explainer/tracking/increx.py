from incremental_explainer.models.base_model import BaseModel
from incremental_explainer.explainers.base_explainer import BaseExplainer
from incremental_explainer.tracking.so_tracker import SoTracker
from incremental_explainer.utils.explanations import compute_initial_sufficient_explanation, compute_subsequent_sufficient_explanation
from torchvision import transforms
import numpy as np

class IncRex:
    
    def __init__(self, model: BaseModel, explainer: BaseExplainer, object_index) -> None:
        self._prev_saliency_map = []
        self._previous_bounding_box = []
        self._frame_number = 0
        self._model = model
        self._explainer = explainer
        self._explanation_tracker = None
        self._object_index = object_index
        self._exp_threshold = 0
    
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
            sufficient_explanation, self._exp_threshold = compute_initial_sufficient_explanation(self._model, saliency_map, image, np.argmax(prediction.class_scores[self._object_index]), bounding_box, divisions=1000)
        else:
            saliency_map, bounding_box = self._explanation_tracker.compute_tracked_explanation(image, prediction)
            sufficient_explanation = compute_subsequent_sufficient_explanation(saliency_map, image, self._exp_threshold)
        
        self._frame_number += 1
        
        self._prev_saliency_map = saliency_map
        return saliency_map, bounding_box, sufficient_explanation