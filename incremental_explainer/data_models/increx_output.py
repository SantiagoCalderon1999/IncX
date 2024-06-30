from pydantic import BaseModel
import numpy as np
from typing import List
from pydantic import BaseModel, Field

class IncRexOutput(BaseModel):
    saliency_map: np.ndarray
    bounding_box: List[int]
    sufficient_explanation: np.ndarray
    label: str

    class Config:
        arbitrary_types_allowed = True
