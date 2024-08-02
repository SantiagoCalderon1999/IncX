import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics
from numpy import trapz
from incrementalexplainer.utils.common import calculate_intersection_over_union
import torchvision.transforms as transforms
from incrementalexplainer.dependencies.d_rise.vision_explanation_methods.explanations import common as od_common
from tqdm import tqdm

def compute_insertion(model: od_common.GeneralObjectDetectionModelWrapper, saliency_map, image, class_index, bounding_box, object_index, divisions=100, verbose=False):
    import matplotlib as mpl
    mpl.rcParams["savefig.pad_inches"] = 0
    masks = np.empty([saliency_map.shape[0], saliency_map.shape[1], 3])
    conf_insertion_list = []
    divisions_list_in = []
    minimum = np.min(saliency_map)
    maximum = np.max(saliency_map)
    thresholds = np.linspace(start=minimum, stop=maximum, num=divisions).tolist()
    thresholds = thresholds[::-1]
    im_size = saliency_map.shape[0] * saliency_map.shape[1] * 3
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    initital_confidence = model.predict([transform(image)])[0].class_scores[object_index][class_index]
    print(initital_confidence)
    for threshold in tqdm(thresholds):
        masks[:, :, :] = False
        pixels = np.where(saliency_map >= threshold)
        masks[pixels[0], pixels[1], :] = True
        div = len(np.where(masks)[0]) / im_size
        divisions_list_in.append(div)
        min_expl = np.where(masks, image, 0)
        img_t = transform(min_expl)
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

        conf_insertion_list.append(float(max_confidence/initital_confidence))
    auc = trapz(conf_insertion_list, divisions_list_in)
    if verbose:
        sns.set_theme(style="whitegrid")

        plt.figure(figsize=(10, 6))
        sns.lineplot(x=divisions_list_in, y=conf_insertion_list)

        plt.fill_between(divisions_list_in, conf_insertion_list, alpha=0.3)

        plt.title(f'Insertion Curve - AUC = {auc:0.4f}', fontsize=32)
        plt.xlabel('Pixels Inserted', fontsize=28)
        plt.ylabel('Confidence', fontsize=28)
        plt.tick_params(axis='both', which='major', labelsize=24)

        # plt.show()

    return auc
