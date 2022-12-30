import pandas as pd
import numpy as np

def euclidean_distance(x,y):
    """
    It computes the euclidean distance of a point(x) to a set of points y.
    distance_y1n = sqrt((x1 - y11)^2 + (x2 - y12)^2) + ... + sqrt((xn - y1n)^2)
    distance_y2n = sqrt((x1 - y21)^2 + (x2 - y22)^2) + ... + sqrt((xn - y2n)^2)
    :param x: Vector of points
    :type x: np.ndarray
    :param y: Set of points
    :type y: np.ndarray
    :return: Euclidean distance for each point in y
    :rtype: np.ndarray
    """
    return np.sqrt(np.sum((x - y) ** 2, axis=1))