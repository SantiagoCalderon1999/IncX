import numpy as np
from incremental_explainer.models.base_model import BaseModel
import matplotlib.pyplot as plt
from incremental_explainer.utils.common import calculate_intersection_over_union
import torchvision.transforms as transforms
    

def compute_initial_sufficient_explanation(model: BaseModel, saliency_map, image, class_index, bounding_box, divisions=1000):
    masks = np.zeros((saliency_map.shape[0], saliency_map.shape[1], 3), dtype=bool)

    minimum = np.min(saliency_map) - np.abs(
        (np.min(saliency_map) - np.max(saliency_map)) * 0.2
    )
    maximum = np.max(saliency_map) + np.abs(
        (np.min(saliency_map) - np.max(saliency_map)) * 0.1
    )

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    for sub_index in range(divisions):
        masks.fill(False)
        threshold = maximum + (sub_index / divisions) * (minimum - maximum)
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

        if max_confidence > 0.3:
            return suf_expl, threshold

    return np.zeros_like(image), threshold
    
def compute_subsequent_sufficient_explanation(saliency_map, image, threshold):
    import matplotlib as mpl
    mpl.rcParams["savefig.pad_inches"] = 0
    masks = np.empty([saliency_map.shape[0], saliency_map.shape[1], 3])

    masks[:, :, :] = False
    pixels = np.where(saliency_map >= threshold)
    masks[pixels[0], pixels[1], :] = True
    suf_expl = np.where(masks, image, 0)

    return suf_expl