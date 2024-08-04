import numpy as np


def compute_explanation_proportion(mask):
    return np.sum(mask) / (mask.shape[0] * mask.shape[1])
