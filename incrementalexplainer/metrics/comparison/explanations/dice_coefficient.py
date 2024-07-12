import numpy as np

def compute_dice_coefficient(mask_1, mask_2):
    intersection = np.logical_and(mask_1, mask_2)
    return 2 * intersection.sum() / (mask_1.sum() + mask_2.sum())