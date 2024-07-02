import numpy as np
from incremental_explainer.models.base_model import BaseModel
import matplotlib.pyplot as plt
from incremental_explainer.utils.common import calculate_intersection_over_union
import torchvision.transforms as transforms
    

def compute_initial_sufficient_explanation(model: BaseModel, saliency_map, image, class_index, bounding_box, divisions=100, minimum=None, maximum=None):
    masks = np.zeros((saliency_map.shape[0], saliency_map.shape[1], 3), dtype=bool)

    if minimum is None:
        minimum = np.min(saliency_map)
        
    if maximum is None:
        maximum = np.max(saliency_map)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    thresholds = np.linspace(start=minimum, stop=maximum, num=divisions).tolist()
    left = 0
    right = len(thresholds) - 1
    
    if minimum is None and maximum is None:
        final_suf_expl = np.zeros_like(image)
        final_threshold = minimum
    else:
        final_threshold = (minimum + maximum) / 2
        masks.fill(False)
        pixels = np.where(saliency_map >= final_threshold)
        masks[pixels[0], pixels[1], :] = True
        final_suf_expl = np.where(masks, image, 0)
        
    while left < right:
        t_index = (left + right) // 2
        threshold = thresholds[t_index]
        masks.fill(False)
        pixels = np.where(saliency_map >= threshold)
        masks[pixels[0], pixels[1], :] = True

        suf_expl = np.where(masks, image, 0)
        img_t = transform(suf_expl).unsqueeze(0)
        detection = model.predict(img_t)
        
        arrays = []
        for i, bbox in enumerate(detection[0].bounding_boxes.cpu().detach()):
            index = np.argmax(detection[0].class_scores[i].cpu().detach())
            if index == class_index:
                arrays.append((detection[0].class_scores[i][index].item(), bbox))

        if arrays:
            max_confidence = max([
                score * calculate_intersection_over_union(bounding_box, bbox)
                for score, bbox in arrays
            ])
        else:
            max_confidence = 0

        if max_confidence > 0:
            left = t_index + 1
            final_suf_expl = suf_expl
            final_threshold = threshold
        else:
            right = t_index - 1
    
    return final_suf_expl, final_threshold
    
def compute_subsequent_sufficient_explanation(saliency_map, image, threshold):
    masks = np.empty([saliency_map.shape[0], saliency_map.shape[1], 3])

    masks[:, :, :] = False
    pixels = np.where(saliency_map >= threshold)
    masks[pixels[0], pixels[1], :] = True
    suf_expl = np.where(masks, image, 0)

    return suf_expl