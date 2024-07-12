from scipy import stats
import numpy as np


def normalize(saliency_map):
    total = np.sum(saliency_map)
    return saliency_map / total

def compute_kullback_leibler_divergence(P, Q):
    epsilon = 1e-10
    P = np.clip(P, epsilon, 1)
    Q = np.clip(Q, epsilon, 1)
    return np.sum(P * np.log(P / Q))