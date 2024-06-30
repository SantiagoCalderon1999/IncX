from incremental_explainer.models.base_model import BaseModel
from incremental_explainer.explainers.base_explainer import BaseExplainer
from incremental_explainer.tracking.exp_tracker import ExpTracker
from incremental_explainer.utils.explanations import compute_initial_sufficient_explanation, compute_subsequent_sufficient_explanation
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from typing import List
from incremental_explainer.data_models.increx_output import IncRexOutput
from collections import defaultdict
from typing import Dict

class IncRex:
    
    def __init__(self, model: BaseModel, explainer: BaseExplainer, object_indices: List[int] = None) -> None:
        self._frame_number = 0
        self._model = model
        self._explainer = explainer
        self._explanation_tracker = None
        self._object_indices = set(object_indices) if object_indices else None
        self._exp_thresholds = defaultdict(int)
    
    def explain_frame(self, image) -> Dict[int, IncRexOutput]:
        
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_t = transform(image)

        prediction = self._model.predict([img_t])
        prediction = prediction[0]
        results = defaultdict(IncRexOutput)
        if self._frame_number == 0:
            
            if not self._object_indices:
                self._object_indices = set(range(len(prediction.bounding_boxes)))
                
            if min(self._object_indices) < 0:
                raise ValueError("One of the provided values in object_indexes is less than zero.")

            if len(prediction.bounding_boxes) <= max(self._object_indices):
                raise ValueError(
                    f"One of the indices provided ({max(self._object_indices)}) is larger than or equal to the number of objects "
                    f"found by the object detector ({len(prediction.bounding_boxes)})."
                )
                            
            saliency_maps = self._explainer.create_saliency_map(prediction, image)
            
            saliency_maps_dict = defaultdict(np.array)
            bounding_boxes_dict = defaultdict(np.array)
            for object_index in self._object_indices:
                saliency_maps_dict[object_index] = saliency_maps[object_index]
                bounding_boxes_dict[object_index] = prediction.bounding_boxes[object_index]

            self._explanation_tracker = ExpTracker(saliency_maps_dict, bounding_boxes_dict, prediction)
            
            for object_index in self._object_indices:
                bounding_box = prediction.bounding_boxes[object_index]
                sufficient_explanation, exp_threshold = compute_initial_sufficient_explanation(self._model, saliency_maps[object_index], image, np.argmax(prediction.class_scores[object_index]), bounding_box)
                self._exp_thresholds[object_index] = exp_threshold
                bounding_box = (int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3]))
                results[object_index] = IncRexOutput(saliency_map=saliency_maps[object_index], bounding_box=bounding_box, sufficient_explanation=sufficient_explanation)

        else:
            tracking_results = self._explanation_tracker.compute_tracked_explanation(image, prediction)
            for object_index, (saliency_map, bounding_box) in tracking_results.items():
                sufficient_explanation = compute_subsequent_sufficient_explanation(saliency_map, image, self._exp_thresholds[object_index])
                results[object_index] = IncRexOutput(saliency_map=saliency_map, bounding_box=bounding_box, sufficient_explanation=sufficient_explanation)
        
        self._frame_number += 1
        return results
    
    def explain_frame_sequence(self, image_set):
        alpha = 0.5
        light_red = (100, 28, 30)
        frames = []
        for image in tqdm(image_set, position=0, leave=True):
            results = self.explain_frame(image)
            object_frames = []
            for el in results.values():
                viridis_frame = plt.cm.viridis(el.saliency_map)
                viridis_frame_rgb = viridis_frame[:, :, :3]
                frame = cv2.addWeighted(
                    image, alpha, (viridis_frame_rgb * 255).astype(np.uint8), 1 - alpha, 0
                )
                frame = cv2.rectangle(frame, (int(el.bounding_box[0]), int(el.bounding_box[1])), (int(el.bounding_box[2]), int(el.bounding_box[3])), light_red, thickness=3)
                frame = np.hstack((frame, el.sufficient_explanation))
                object_frames.append(frame)
            current_frame = np.vstack(object_frames)
            frames.append(current_frame)
        
        return frames
    
    def explain_video(self, video_path):
        vid_obj  = cv2.VideoCapture(video_path)
        frames = []
        success, frame = vid_obj.read()
        while success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            success, frame = vid_obj.read()

        vid_obj.release()
        
        return self.explain_frame_sequence(frames)