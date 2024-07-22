from incrementalexplainer.tracking.increx  import IncRex
from incrementalexplainer.models.model_enum import ModelEnum
from incrementalexplainer.models.model_factory import ModelFactory
from incrementalexplainer.explainers.d_rise import DRise
import numpy as np
from PIL import Image
import cv2
from incrementalexplainer.metrics.saliency_maps.epg import compute_energy_based_pointing_game

def test_epg_value():
    
    # Given
    image_locations = [f'datasets/LASOT/1/{str(i).zfill(8)}.jpg' for i in range(1, 10)]
    images = [resize_image(image_location, (640, 480)) for image_location in image_locations]
    model = ModelFactory().get_model(ModelEnum.YOLO)
    explainer = DRise(model, 500)
    incRex = IncRex(model, explainer)

    # When
    average_epg = 0
    for i, image in enumerate(images):
        results,_ = incRex.explain_frame(image)
        result = results[0]
        epg = compute_energy_based_pointing_game(result.saliency_map, result.bounding_box)
        if i != 0:
            average_epg += epg
    average_epg = average_epg / (len(images)-1)
    
    # Then
    assert average_epg < 0.1

def resize_image(image_path, target_size):
    pil_image = Image.open(image_path)
    image_array = np.array(pil_image)
    cv2_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    resized_image = cv2.resize(cv2_image, target_size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
