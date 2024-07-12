import numpy as np

def compute_scanpath_saliency(saliency_1, saliency_2):
    """
    Compute the Normalized Scanpath Saliency (NSS) between two saliency maps.
    
    Parameters:
    saliency_map (ndarray): Predicted saliency map as a 2D numpy array.
    fixation_map (ndarray): Binary fixation map as a 2D numpy array where 1 indicates a fixation and 0 indicates no fixation.
    
    Returns:
    float: The NSS value.
    """
    
    # Calculate mean and standard deviation of the saliency map
    mean_saliency = np.mean(saliency_1)
    std_saliency = np.std(saliency_1)
    
    # Normalize the saliency map
    normalized_saliency = (saliency_1 - mean_saliency) / std_saliency
    
    # Calculate the NSS
    nss = np.sum(normalized_saliency * saliency_2) / np.sum(saliency_2)
    
    return nss