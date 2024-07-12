def calculate_intersection_over_union(square1, square2):
    x1 = max(square1[0], square2[0])
    y1 = max(square1[1], square2[1])
    x2 = min(square1[2], square2[2])
    y2 = min(square1[3], square2[3])

    area_square1 = (square1[2] - square1[0]) * (square1[3] - square1[1])
    area_square2 = (square2[2] - square2[0]) * (square2[3] - square2[1])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    union_area = area_square1 + area_square2 - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0

    return iou