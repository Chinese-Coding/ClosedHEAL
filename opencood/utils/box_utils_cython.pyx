import numpy as np
cimport numpy as np

def ProjectPointsByMatrixUsingCython32(np.ndarray[np.float32_t, ndim=2] points,
                                     np.ndarray[np.float32_t, ndim=2] transformation_matrix):

    # convert to homogeneous coordinates via padding 1 at the last dimension
    cdef np.ndarray[np.float32_t, ndim=2] points_homogeneous = np.pad(points, ((0, 0), (0, 1)), mode="constant", constant_values=1)

    # 使用 numpy 的高效矩阵乘法
    cdef np.ndarray[np.float32_t, ndim=2] projected_points_homogeneous = np.dot(points_homogeneous, transformation_matrix.T)

    return projected_points_homogeneous[:, :3]

def ProjectPointsByMatrixUsingCython64(np.ndarray[np.float64_t, ndim=2] points,
                                     np.ndarray[np.float64_t, ndim=2] transformation_matrix):

    # convert to homogeneous coordinates via padding 1 at the last dimension
    cdef np.ndarray[np.float64_t, ndim=2] points_homogeneous = np.pad(points, ((0, 0), (0, 1)), mode="constant", constant_values=1)

    # 使用 numpy 的高效矩阵乘法
    cdef np.ndarray[np.float64_t, ndim=2] projected_points_homogeneous = np.dot(points_homogeneous, transformation_matrix.T)

    return projected_points_homogeneous[:, :3]