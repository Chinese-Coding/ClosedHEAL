import numpy as np
cimport numpy as cnp



def ShufflePoints(cnp.ndarray[cnp.float32_t, ndim=2] points):
    return points[np.random.permutation(points.shape[0])]

def MaskEgoPoints(cnp.ndarray[cnp.float32_t, ndim=2] points):
    return points[np.logical_not(
        points[:, 0] >= -1.95) & (points[:, 0] <= 2.95) & (points[:, 1] >= -1.1) & (points[:, 1] <= 1.1
    )]