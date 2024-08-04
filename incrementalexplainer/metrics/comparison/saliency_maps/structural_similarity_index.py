from skimage.metrics import structural_similarity as ssim


def compute_structural_similarity_index(saliency_1, saliency_2):
    return ssim(saliency_1, saliency_2, data_range=saliency_2.max() - saliency_2.min())
