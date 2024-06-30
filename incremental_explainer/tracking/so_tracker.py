from sort import *
from sort import *
import numpy as np

from incremental_explainer.transformations.image_scaling import scale_image
from incremental_explainer.transformations.image_moving import move_image
from vision_explanation_methods.explanations import common as od_common
class SoTracker:
    
    def __init__(self, initial_saliency_map, initial_prediction: od_common.DetectionRecord, object_index):
        self._initial_saliency_map = initial_saliency_map
        self._initial_bounding_box = initial_prediction.bounding_boxes[object_index]
        self._tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        detections = [[float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]), float(max(score))] for bb, score in zip(initial_prediction.bounding_boxes, initial_prediction.class_scores)]
        detections = np.array(detections)
        tracked = self._tracker.update(detections)
        self._id = 0
        for tracked_object in tracked:
            if tracked_object[0] == self._initial_bounding_box[0] and tracked_object[1] == self._initial_bounding_box[1] and tracked_object[2] == self._initial_bounding_box[2] and tracked_object[3] == self._initial_bounding_box[3]:
                self._id = tracked_object[4]
    
    def compute_tracked_explanation(self, image, prediction: od_common.DetectionRecord):
        detections = [
            [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]), float(max(score))]
            for bb, score in zip(prediction.bounding_boxes, prediction.class_scores)
        ]

        if not detections:
            return np.zeros_like(self._initial_saliency_map), self._initial_bounding_box

        detections = np.array(detections)
        result_tracker = self._tracker.update(detections)

        matching_result = next((result for result in result_tracker if result[4] == self._id), np.array([]))

        if not matching_result.any():
            return np.zeros_like(self._initial_saliency_map), [0, 0, 0, 0]

        x1_new, y1_new, x2_new, y2_new = map(int, matching_result[:4])
        x1_old, y1_old, x2_old, y2_old = self._initial_bounding_box

        old_width, old_height = abs(x1_old - x2_old), abs(y1_old - y2_old)
        new_width, new_height = abs(x1_new - x2_new), abs(y1_new - y2_new)

        scale_x = new_width / old_width
        scale_y = new_height / old_height

        scaled_saliency_map = scale_image(self._initial_saliency_map, scale_x, scale_y)

        old_bb_center = np.array([(x1_old + x2_old) / 2, (y1_old + y2_old) / 2])
        new_bb_center = np.array([(x1_new + x2_new) / 2, (y1_new + y2_new) / 2])

        image_center = np.array([image.shape[1] / 2, image.shape[0] / 2])
        scaled_map_center = np.array([scaled_saliency_map.shape[1] / 2, scaled_saliency_map.shape[0] / 2])

        scaled_bb_center = np.array([
            (old_bb_center[0] - image_center[0]) * scale_x + scaled_map_center[0],
            (old_bb_center[1] - image_center[1]) * scale_y + scaled_map_center[1]
        ])

        center_changes = new_bb_center - scaled_bb_center

        expanded_saliency_map = move_image(
            np.transpose(scaled_saliency_map, (0, 1)),
            int(center_changes[0]),
            int(center_changes[1]),
            (image.shape[0], image.shape[1]),
        )

        return expanded_saliency_map, (x1_new, y1_new, x2_new, y2_new)
