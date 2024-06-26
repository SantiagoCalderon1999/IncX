
from vision_explanation_methods.explanations import common as od_common
import numpy as np
import torch

from ultralytics import YOLO

import numpy as np

class Yolo(od_common.GeneralObjectDetectionModelWrapper):
    """Wraps a PytorchFasterRCNN model with a predict API function for object detection.

    To be compatible with the drise explainability method, all models must be wrapped to have
    the same output and input class.
    This wrapper is customized for the FasterRCNN model from Pytorch, and can
    also be used with the RetinaNet or any other models with the same output class.
    """
    
    def __init__(self):
        self._model = YOLO("yolov10n.pt")
        self._number_of_classes = 80

    def predict(self, x: torch.tensor):
        """Creates a list of detection records from the image predictions.
        """
        raw_detections = []
        for x_el in x:
            input  = (np.ascontiguousarray(np.transpose(x_el.cpu().numpy(), (1, 2, 0))) * 255).astype('uint8')
            raw_detection = self._model.predict(input, verbose = False)
            raw_detections.append(raw_detection)
        
        detections = [] 
        for raw_detection in raw_detections:
            scores = raw_detection[0].boxes.conf
            boxes = raw_detection[0].boxes.xyxy
            class_ids = raw_detection[0].boxes.cls
            
            indices = scores > 0.5
            
            scores = scores[indices]
            boxes = boxes[indices]
            class_ids = class_ids[indices]
            
            detections.append(
                od_common.DetectionRecord(
                    bounding_boxes=boxes,
                    class_scores=od_common.expand_class_scores(scores,
                                                                class_ids,
                                                                self._number_of_classes),
                    objectness_scores=torch.tensor([1.0]*len(scores)),
                )
            )
        
        return detections
