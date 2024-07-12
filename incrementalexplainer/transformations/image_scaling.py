import cv2

def scale_image(image, scale_x, scale_y):
    width = int(image.shape[1] * scale_x)
    height = int(image.shape[0] * scale_y)
    dim = (width, height)
    scaled_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return scaled_image