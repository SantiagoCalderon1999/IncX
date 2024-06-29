from sort import *
import cvzone
from sort import *
import math
import numpy as np
import cv2

import cv2
import matplotlib.pyplot as plt
from incremental_explainer.transformations.image_scaling import scale_image
from incremental_explainer.transformations.image_moving import move_image
from incremental_explainer.utils.common import calculate_intersection_over_union
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
        print('ID: ', self._id)
    
    def compute_tracked_explanation(self, image, prediction: od_common.DetectionRecord):
        detections = [[float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]), float(max(score))] for bb, score in zip(prediction.bounding_boxes, prediction.class_scores)]
        if (len(detections) == 0):
            return (np.zeros_like(self._initial_saliency_map), self._initial_bounding_box)
        detections = np.array(detections)
        resultTracker = self._tracker.update(detections)
        center_changes = {}
        height = image.shape[0]
        width = image.shape[1]
        
        for i, res in enumerate(resultTracker):
            x1, y1, x2,y2, id = res
            x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
            matching_index_x = 0
            matching_box = []
            if (self._id == id):
                x1_old, y1_old, x2_old, y2_old = self._initial_bounding_box
                best_iou = 0
                for i, detection in enumerate(detections):
                    iou = calculate_intersection_over_union((detection[0], detection[1], detection[2], detection[3]), (x1, y1, x2, y2))
                    if iou > best_iou:
                        matching_box = detection
                        matching_index_x = i
                #confidence = confidence_dict[(matching_box[0], matching_box[1], matching_box[2], matching_box[3])]
                # print(f'x_old: {x1_old}, y_old: {y1_old}, x2_old: {x2_old}, y2_old: {y2_old}')
                old_x = abs(x1_old - x2_old)
                old_y = abs(y1_old - y2_old)
                new_x = abs(x1 - x2)
                new_y = abs(y1 - y2)

                scale_x = new_x / old_x
                scale_y = new_y / old_y
                exp_scaled = scale_image(self._initial_saliency_map, scale_x, scale_y)

                old_bb_center = np.array([(x1_old + x2_old) / 2, (y1_old + y2_old) / 2])
                new_bb_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

                old_image_center = np.array([width / 2, height / 2])
                scaled_image_center = np.array([exp_scaled.shape[1] / 2, exp_scaled.shape[0] / 2])
                scaled_bb_center = np.array([(old_bb_center[0] - old_image_center[0]) * scale_x + scaled_image_center[0], (old_bb_center[1] - old_image_center[1]) * scale_y + scaled_image_center[1]])

                center_changes = new_bb_center - scaled_bb_center

                exp = move_image(
                    np.transpose(exp_scaled, (0, 1)),
                    int(center_changes[0]),
                    int(center_changes[1]),
                    (height, width),
                )
                
                # cvzone.putTextRect(
                #     image,
                #     text=f"{class_name}:{confidence}, id: {first_id}",
                #     pos=(x1 + 10, y1 - 10),
                #     scale=1.5,
                #     thickness=2,
                #     colorR=light_blue,
                #     font=cv2.FONT_HERSHEY_PLAIN,
                # )
                
                return (exp, (x1, y1, x2, y2))
            
        return (np.zeros_like(self._initial_saliency_map), self._initial_bounding_box)