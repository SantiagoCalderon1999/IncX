import numpy as np
import numpy as np


def move_image(image, delta_x, delta_y, dimensions):
    height, width = image.shape[:2]
    new_image = np.zeros((dimensions[0], dimensions[1]))
    new_x_start, new_y_start, new_x_end, new_y_end = _calculate_position(delta_x, delta_y, dimensions, height, width)

    orig_x_start, orig_y_start = max(0, -delta_x), max(0, -delta_y)
    orig_x_end, orig_y_end = min(width, dimensions[1] - delta_x), min(
        height, dimensions[0] - delta_y
    )

    new_image[new_y_start:new_y_end, new_x_start:new_x_end] = image[
        orig_y_start:orig_y_end, orig_x_start:orig_x_end
    ]

    return new_image

def _calculate_position(delta_x, delta_y, dimensions, height, width):
    new_x_start, new_y_start = max(0, delta_x), max(0, delta_y)
    new_x_end, new_y_end = min(dimensions[1], width + delta_x), min(
        dimensions[0], height + delta_y
    )
    
    return new_x_start,new_y_start,new_x_end,new_y_end