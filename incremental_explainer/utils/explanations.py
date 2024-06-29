import numpy as np
from incremental_explainer.models.base_model import BaseModel
import matplotlib.pyplot as plt
from incremental_explainer.utils.common import calculate_intersection_over_union
import torchvision.transforms as transforms
    

def compute_initial_sufficient_explanation(model: BaseModel, saliency_map, image, class_index, bounding_box, divisions=100):
        import matplotlib as mpl
        mpl.rcParams["savefig.pad_inches"] = 0
        masks = np.empty([saliency_map.shape[0], saliency_map.shape[1], 3])

        minimum = np.min(saliency_map) - np.abs(
            (np.min(saliency_map) - np.max(saliency_map)) * 0.2
        )
        maximum = np.max(saliency_map) + np.abs(
            (np.min(saliency_map) - np.max(saliency_map)) * 0.1
        )

        for sub_index in range(0, divisions):
            masks[:, :, :] = False
            threshold = maximum + (sub_index / divisions) * (minimum - maximum)
            pixels = np.where(saliency_map >= threshold)
            masks[pixels[0], pixels[1], :] = True

            suf_expl = np.where(masks, image, 0)
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            img_t = transform(suf_expl)
            detection = model.predict([img_t])            
            arrays = []
            
            for i, _ in enumerate(detection[0].bounding_boxes.cpu().detach()):
                index = np.argmax(detection[0].class_scores[i].cpu().detach())
                if index == class_index:
                    arrays.append((detection[0].class_scores[i][index].cpu().detach(), detection[0].bounding_boxes[i].cpu().detach()))
            if len(arrays) > 0:
                max_confidence = max([el[0] * calculate_intersection_over_union(bounding_box, el[1]) for el in arrays])
            else:
                max_confidence = 0

            if (max_confidence > 0.5):

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