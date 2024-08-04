from incrementalexplainer.models.model_enum import ModelEnum
from incrementalexplainer.models.base_model import BaseModel
from incrementalexplainer.models.rt_detr import RtDetr
from incrementalexplainer.models.yolo import Yolo
from incrementalexplainer.models.faster_rcnn import FasterRcnn


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
