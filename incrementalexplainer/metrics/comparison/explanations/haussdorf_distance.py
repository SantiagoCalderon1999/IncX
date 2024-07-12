import numpy as np
from scipy.spatial.distance import directed_hausdorff


def compute_haussdorf_distance(mask_1, mask_2):    

    mask_1_coords = np.argwhere(mask_1)
    mask_2_coords = np.argwhere(mask_2)


    distance_1 = directed_hausdorff(mask_1_coords, mask_2_coords)[0]
    distance_2 = directed_hausdorff(mask_2_coords, mask_1_coords)[0]

    return max(distance_1, distance_2)