from scipy import stats
import numpy as np


def compute_pearson_coefficient(array_1, array_2):
    result = stats.pearsonr(np.array(array_1).flatten(), np.array(array_2).flatten())
    return result.statistic
