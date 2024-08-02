from incrementalexplainer.models.base_model import BaseModel
from incrementalexplainer.explainers.base_explainer import BaseExplainer
from incrementalexplainer.tracking.exp_tracker import ExpTracker
from incrementalexplainer.utils.explanations import compute_initial_sufficient_explanation, compute_subsequent_sufficient_explanation
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from typing import List
from incrementalexplainer.data_models.increx_output import IncRexOutput
from collections import defaultdict
from typing import Dict
import cvzone
from incrementalexplainer.models.labels import coco_labels


class IncRex:
    
    def __init__(self, model: BaseModel, explainer: BaseExplainer, object_indices: List[int] = None, saliency_map_divisions: int = 100) -> None:
        self._frame_number = 0
        self._model = model
        self._explainer = explainer
        self._explanation_tracker = None
        self._object_indices = set(object_indices) if object_indices else None
        self._exp_thresholds = defaultdict(int)
        self._obj_classes_ix = defaultdict(int)
        self._saliency_map_divisions = saliency_map_divisions
    
    def explain_frame(self, image) -> tuple[Dict[int, IncRexOutput], np.ndarray]:
        
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_t = transform(image)

        prediction = self._model.predict([img_t])
        prediction = prediction[0]
        results = defaultdict(IncRexOutput)
        if self._frame_number == 0:
            
            if len(prediction.bounding_boxes) == 0:
                raise ValueError("No objects were detected in the first frame.")
            
            if not self._object_indices:
                self._object_indices = set(range(len(prediction.bounding_boxes)))
                
            if min(self._object_indices) < 0:
                raise ValueError("One of the provided values in object_indexes is less than zero.")

            if len(prediction.bounding_boxes) <= max(self._object_indices):
                raise ValueError(
                    f"One of the indices provided ({max(self._object_indices)}) is larger than or equal to the number of objects "
                    f"found by the object detector ({len(prediction.bounding_boxes)})."
                )
                            
            saliency_maps = self._explainer.create_saliency_map(image)
            
            saliency_maps_dict = defaultdict(np.array)
            bounding_boxes_dict = defaultdict(np.array)
            for object_index in self._object_indices:
                saliency_maps_dict[object_index] = saliency_maps[object_index]
                bounding_boxes_dict[object_index] = prediction.bounding_boxes[object_index]

            self._explanation_tracker = ExpTracker(saliency_maps_dict, bounding_boxes_dict, prediction)

            for object_index in self._object_indices:
                bounding_box = prediction.bounding_boxes[object_index]
                self._obj_classes_ix[object_index] = np.argmax(prediction.class_scores[object_index])
                sufficient_explanation, exp_threshold, mask = compute_initial_sufficient_explanation(self._model, saliency_maps[object_index], image, self._obj_classes_ix[object_index], bounding_box, divisions=self._saliency_map_divisions, minimum=np.min(saliency_maps[object_index]), maximum=np.max(saliency_maps[object_index]))
                self._exp_thresholds[object_index] = exp_threshold
                bounding_box = (int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3]))
                results[object_index] = IncRexOutput(saliency_map=saliency_maps[object_index], bounding_box=bounding_box, sufficient_explanation=sufficient_explanation, label=coco_labels[self._obj_classes_ix[object_index]], score=float(max(prediction.class_scores[object_index])), mask=mask, current_index=object_index)

        else:
            tracking_results = self._explanation_tracker.compute_tracked_explanation(image, prediction)
            for object_index, (saliency_map, bounding_box, score, current_index) in tracking_results.items():
                if score > 0:
                    sufficient_explanation, exp_threshold, mask = compute_initial_sufficient_explanation(self._model, saliency_map, image, self._obj_classes_ix[object_index], bounding_box, divisions=5, minimum= self._exp_thresholds[object_index] * 0.8, maximum= self._exp_thresholds[object_index] * 1.2)
                    self._exp_thresholds[object_index] = exp_threshold
                else:
                    sufficient_explanation = np.zeros_like(image)
                    mask = np.zeros_like(saliency_map)
                results[object_index] = IncRexOutput(saliency_map=saliency_map, bounding_box=bounding_box, sufficient_explanation=sufficient_explanation, label=coco_labels[self._obj_classes_ix[object_index]], score=score, mask=mask, current_index=current_index)
        
        self._frame_number += 1
        
        bright_red = (255, 0, 64)
        alpha = 0.5
        object_frames = []
        for object_index, el in results.items():
            viridis_frame = plt.cm.jet(el.saliency_map)
            viridis_frame_rgb = viridis_frame[:, :, :3]
            frame = cv2.addWeighted(
                image, alpha, (viridis_frame_rgb * 255).astype(np.uint8), 1 - alpha, 0
            )
            frame = cv2.rectangle(frame, (int(el.bounding_box[0]), int(el.bounding_box[1])), (int(el.bounding_box[2]), int(el.bounding_box[3])), bright_red, thickness=3)
            el.sufficient_explanation = cv2.rectangle(el.sufficient_explanation, (int(el.bounding_box[0]), int(el.bounding_box[1])), (int(el.bounding_box[2]), int(el.bounding_box[3])), bright_red, thickness=3)
            cvzone.putTextRect(
                frame,
                text=f"{el.label}: {el.score:.2f} ix: {object_index}",
                pos=(el.bounding_box[0] + 8, el.bounding_box[1] - 10),
                scale=1.5,
                thickness=2,
                colorR=bright_red,
                font=cv2.FONT_HERSHEY_DUPLEX,
            )
            frame = np.hstack((frame, el.sufficient_explanation))
            object_frames.append(frame)
        current_frame = np.vstack(object_frames)
        return results, current_frame
    
    def explain_frame_sequence(self, image_set):
        frames = []
        for image in tqdm(image_set, position=0):
            _, current_frame = self.explain_frame(image)
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