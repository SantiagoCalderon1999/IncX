from incrementalexplainer.tracking.increx  import IncRex
from incrementalexplainer.models.model_enum import ModelEnum
from incrementalexplainer.models.model_factory import ModelFactory
from incrementalexplainer.explainers.d_rise import DRise
import numpy as np
import time
from PIL import Image
import cv2

def test_single_image_support():
    
    # Given
    image_location = f'datasets/LASOT/1/{str(1).zfill(8)}.jpg'
    image = resize_image(image_location, (640, 480))
    model = ModelFactory().get_model(ModelEnum.YOLO)
    explainer = DRise(model, 100)
    incRex = IncRex(model, explainer)

    # When
    result = incRex.explain_frame(image)

    # Then
    assert result is not None

    
    
def resize_image(image_path, target_size):
    pil_image = Image.open(image_path)
    image_array = np.array(pil_image)
    cv2_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    resized_image = cv2.resize(cv2_image, target_size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
