from dependencies.sort.sort import *
import numpy as np

from incrementalexplainer.transformations.image_scaling import scale_image
from incrementalexplainer.transformations.image_moving import move_image
from vision_explanation_methods.explanations import common as od_common
from collections import defaultdict

class ExpTracker:
    
    def __init__(self, initial_saliency_maps: dict, initial_bounding_boxes: dict, initial_prediction: od_common.DetectionRecord):
        if (len(initial_saliency_maps) != len(initial_bounding_boxes)):
            raise ValueError("The given saliency maps and boxes do not have the same length")
        
        self._initial_saliency_maps = initial_saliency_maps
        self._initial_bounding_boxes = initial_bounding_boxes
        self._tracker = Sort(max_age=500, min_hits=3, iou_threshold=0.3)
        detections = [[float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]), float(max(score))] for bb, score in zip(initial_prediction.bounding_boxes, initial_prediction.class_scores)]
        detections = np.array(detections)
        tracked = self._tracker.update(detections)
        self._object_to_ids = defaultdict(int)
        bboxes_to_id = {}
        for tracked_tuple in tracked:
            bboxes_to_id[(int(tracked_tuple[0]), int(tracked_tuple[1]), int(tracked_tuple[2]), int(tracked_tuple[3]))] = tracked_tuple[4]
            
        for object_index, bounding_box in self._initial_bounding_boxes.items():                
                self._object_to_ids[object_index] = bboxes_to_id[(int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3]))]

    def compute_tracked_explanation(self, image, prediction: od_common.DetectionRecord):
        detections = [
            [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]), float(max(score))]
            for bb, score in zip(prediction.bounding_boxes, prediction.class_scores)
        ]
        results = {}
        if not detections:
            for object_index in self._object_to_ids.keys():
                results[object_index] = np.zeros_like(self._initial_saliency_maps[object_index]), (0, 0, 0, 0), 0
            return results

        detections = np.array(detections)
        result_tracker = self._tracker.update(detections)

        for object_index, id in self._object_to_ids.items():
            matching_result = next((result for result in result_tracker if result[4] == id), np.array([]))
            if len(matching_result) == 0:
                results[object_index] = np.zeros_like(self._initial_saliency_maps[object_index]), (0, 0, 0, 0), 0
                continue

            x1_new, y1_new, x2_new, y2_new = map(int, matching_result[:4])
            x1_old, y1_old, x2_old, y2_old = self._initial_bounding_boxes[object_index].cpu().numpy().astype(int)

            old_width, old_height = abs(x1_old - x2_old), abs(y1_old - y2_old)
            new_width, new_height = abs(x1_new - x2_new), abs(y1_new - y2_new)

            scale_x = new_width / old_width
            scale_y = new_height / old_height

            scaled_saliency_map = scale_image(self._initial_saliency_maps[object_index], scale_x, scale_y)

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
            results[object_index] = expanded_saliency_map, (x1_new, y1_new, x2_new, y2_new), matching_result[5]

        return results
