from vision_explanation_methods.explanations import common as od_common
import torch, torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from incremental_explainer.models.base_model import BaseModel

class FasterRcnn(BaseModel):
    """Wraps a PytorchFasterRCNN model with a predict API function for object detection.

    To be compatible with the drise explainability method, all models must be wrapped to have
    the same output and input class.
    This wrapper is customized for the FasterRCNN model from Pytorch, and can
    also be used with the RetinaNet or any other models with the same output class.
    """
    
    def __init__(self):
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        model.eval()
        self._model = model
        self._number_of_classes = 91

    def predict(self, x: torch.Tensor):
        """Creates a list of detection records from the image predictions.
        """
        raw_detections = self._model(x)

        def apply_nms(orig_prediction: dict, iou_thresh: float=0.5):
            """Performs non maximum suppression on the predictions according to their intersection-over-union.
            """
            keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

            nms_prediction = orig_prediction
            nms_prediction['boxes'] = nms_prediction['boxes'][keep]
            nms_prediction['scores'] = nms_prediction['scores'][keep]
            nms_prediction['labels'] = nms_prediction['labels'][keep]
            return nms_prediction
        
        def filter_score(orig_prediction: dict, score_thresh: float=0.5):
            """Filters out model predictions with confidence scores below score_thresh
            """
            keep = orig_prediction['scores'] > score_thresh

            filter_prediction = orig_prediction
            filter_prediction['boxes'] = filter_prediction['boxes'][keep]
            filter_prediction['scores'] = filter_prediction['scores'][keep]
            filter_prediction['labels'] = filter_prediction['labels'][keep]
            return filter_prediction
        
        detections = [] 
        for raw_detection in raw_detections:
            raw_detection = apply_nms(raw_detection,0.005)
            
            # Note that FasterRCNN doesn't return a score for each class, only the predicted class
            # DRISE requires a score for each class. We approximate the score for each class
            # by dividing the (1.0 - class score) evenly among the other classes.
            
            raw_detection = filter_score(raw_detection, 0.5)
            expanded_class_scores = od_common.expand_class_scores(raw_detection['scores'],
                                                                  raw_detection['labels'],
                                                                  self._number_of_classes)
            detections.append(
                od_common.DetectionRecord(
                    bounding_boxes=raw_detection['boxes'],
                    class_scores=expanded_class_scores,
                    objectness_scores=torch.tensor([1.0]*raw_detection['boxes'].shape[0]),
                    
                )
            )
        
        return detections