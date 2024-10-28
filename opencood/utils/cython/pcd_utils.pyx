import numpy as np
cimport numpy as cnp


def ProcessPoints(cnp.ndarray[cnp.float64_t, ndim=2] points):
    points = ShufflePoints(points)
    return MaskEgoPoints(points)

cdef inline cnp.ndarray[cnp.float64_t, ndim=2] ShufflePoints(cnp.ndarray[cnp.float64_t, ndim=2] points):
        return points[np.random.permutation(points.shape[0])]

cdef cnp.ndarray[cnp.float64_t, ndim=2] MaskEgoPoints(cnp.ndarray[cnp.float64_t, ndim=2] points):
    cdef cnp.ndarray[cnp.npy_bool, ndim=1] mask = (points[:, 0] >= -1.95) & (points[:, 0] <= 2.95) & (points[:, 1] >= -1.1) & (points[:, 1] <= 1.1)
    return points[np.logical_not(mask)]
