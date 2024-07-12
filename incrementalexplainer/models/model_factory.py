from incremental_explainer.models.model_enum import ModelEnum
from incremental_explainer.models.base_model import BaseModel
from incremental_explainer.models.rt_detr import RtDetr
from incremental_explainer.models.yolo import Yolo
from incremental_explainer.models.faster_rcnn import FasterRcnn


class ModelFactory:
    
    def get_model(self, model: ModelEnum) -> BaseModel:
        if model == ModelEnum.RT_DETR:
            return RtDetr()
        elif model == ModelEnum.YOLO:
            return Yolo()
        elif model == ModelEnum.FASTER_RCNN:
            return FasterRcnn()
        else:
            raise ValueError(f"Unsupported model identifier: {model}")