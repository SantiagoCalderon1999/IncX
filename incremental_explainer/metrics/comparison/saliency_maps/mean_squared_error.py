from scipy import stats
import numpy as np


def compute_mean_squared_error(array_1, array_2):
    return np.square(np.subtract(np.array(array_1).flatten(),np.array(array_2).flatten())).mean() 