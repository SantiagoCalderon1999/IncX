import numpy as np

def compute_jaccard_index(mask_1, mask_2):
    intersection = np.logical_and(mask_1, mask_2)
    union = np.logical_or(mask_1, mask_2)
    return intersection.sum() / union.sum()