from pydantic import BaseModel
import numpy as np
from typing import List


class IncRexOutput(BaseModel):
    saliency_map: np.ndarray
    bounding_box: List[int]
    sufficient_explanation: np.ndarray
    mask: np.ndarray
    label: str
    score: float
    current_index: int

    class Config:
        arbitrary_types_allowed = True
