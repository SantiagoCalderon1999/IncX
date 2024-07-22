from incrementalexplainer.tracking.increx import IncRex
from incrementalexplainer.models.model_enum import ModelEnum
from incrementalexplainer.models.model_factory import ModelFactory
from incrementalexplainer.explainers.d_rise import DRise
import numpy as np
import cv2
from PIL import Image
import os

def test_consecutive_image_support():
    
    # Given
    image_locations = [f'datasets/LASOT/1/{str(i).zfill(8)}.jpg' for i in range(1, 5)]
    images = [resize_image(image_location, (640, 480)) for image_location in image_locations]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4 files
    video = cv2.VideoWriter('test_video.mp4', fourcc, 30, (640, 480))  # Fixed size to match resized images
    
    for image in images:
        video.write(image)
    video.release()
    
    model = ModelFactory().get_model(ModelEnum.YOLO)
    explainer = DRise(model, 100)
    incRex = IncRex(model, explainer)

    # When
    result = incRex.explain_video('test_video.mp4')
    
    # Then
    assert result is not None
    os.remove('test_video.mp4')

def resize_image(image_path, target_size):
    pil_image = Image.open(image_path)
    image_array = np.array(pil_image)
    cv2_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    resized_image = cv2.resize(cv2_image, target_size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
