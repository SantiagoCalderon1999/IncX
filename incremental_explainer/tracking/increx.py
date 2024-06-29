from incremental_explainer.models.base_model import BaseModel
from incremental_explainer.explainers.base_explainer import BaseExplainer
from incremental_explainer.tracking.so_tracker import SoTracker
from incremental_explainer.utils.explanations import compute_initial_sufficient_explanation, compute_subsequent_sufficient_explanation
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

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
    
    def explain_frame(self, image):
        
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
            sufficient_explanation, self._exp_threshold = compute_initial_sufficient_explanation(self._model, saliency_map, image, np.argmax(prediction.class_scores[self._object_index]), bounding_box)
        else:
            saliency_map, bounding_box = self._explanation_tracker.compute_tracked_explanation(image, prediction)
            sufficient_explanation = compute_subsequent_sufficient_explanation(saliency_map, image, self._exp_threshold)
        
        self._frame_number += 1
        
        self._prev_saliency_map = saliency_map
        return saliency_map, bounding_box, sufficient_explanation
    
    def explain_frame_sequence(self, image_set):
        alpha = 0.5
        light_red = (100, 28, 30)
        frames = []
        for image in tqdm(image_set, position=0, leave=True):
            saliency_maps, bounding_box, suff_explanation = self.explain_frame(image)
            viridis_frame = plt.cm.viridis(saliency_maps)
            viridis_frame_rgb = viridis_frame[:, :, :3]
            frame = cv2.addWeighted(
                    image, alpha, (viridis_frame_rgb * 255).astype(np.uint8), 1 - alpha, 0
                )
            cv2.rectangle(frame, (int(bounding_box[0]), int(bounding_box[1])), (int(bounding_box[2]), int(bounding_box[3])), light_red, thickness=3)
            cv2.rectangle(suff_explanation, (int(bounding_box[0]), int(bounding_box[1])), (int(bounding_box[2]), int(bounding_box[3])), light_red, thickness=3)
            frame = np.hstack((frame, suff_explanation))
            frames.append(frame)
        
        return frames