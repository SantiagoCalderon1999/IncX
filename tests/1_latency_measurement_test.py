from incrementalexplainer.tracking.incx import IncX
from incrementalexplainer.models.model_enum import ModelEnum
from incrementalexplainer.models.model_factory import ModelFactory
from incrementalexplainer.explainers.d_rise import DRise
import numpy as np
import time
from PIL import Image
import cv2


def test_latency_measurement():
    # Given
    image_locations = [f"datasets/LASOT/1/{str(i).zfill(8)}.jpg" for i in range(1, 10)]
    images = [
        resize_image(image_location, (640, 480)) for image_location in image_locations
    ]
    model = ModelFactory().get_model(ModelEnum.YOLO)
    explainer = DRise(model, 100)
    incRex = IncX(model, explainer)

    # When
    average_time = 0
    for i, image in enumerate(images):
        start = time.time()
        incRex.explain_frame(image)
        end = time.time()

        if i != 0:
            average_time += end - start

    average_time = average_time / (len(images) - 1) * 1000

    # Then
    assert average_time < 900


def resize_image(image_path, target_size):
    pil_image = Image.open(image_path)
    image_array = np.array(pil_image)
    cv2_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    resized_image = cv2.resize(cv2_image, target_size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
