from incx.explainers.explainer_enum import ExplainerEnum
from incx.tracking.incx import IncX
from incx.models.model_enum import ModelEnum
from incx.models.model_factory import ModelFactory
from incx.explainers.d_rise import DRise
import numpy as np
from PIL import Image
import cv2


def test_latency_measurement():
    # Given
    image_location = f"datasets/LASOT/1/{str(2).zfill(8)}.jpg"
    image = resize_image(image_location, (640, 480))

    # When
    number_of_explainer = len(ExplainerEnum)
    successful_counter = 0
    for model_name in ModelEnum:
        model = ModelFactory().get_model(model_name)
        explainer = DRise(model, 100)
        incRex = IncX(model, explainer)
        incRex.explain_frame(image)
        successful_counter += 1

    # Then
    assert number_of_explainer > 0
    assert successful_counter == len(ModelEnum)


def resize_image(image_path, target_size):
    pil_image = Image.open(image_path)
    image_array = np.array(pil_image)
    cv2_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    resized_image = cv2.resize(cv2_image, target_size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
