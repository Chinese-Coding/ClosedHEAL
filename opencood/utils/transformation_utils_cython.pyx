import numpy as np
cimport numpy as np

# 定义类型
ctypedef np.float64_t DTYPE_t

cdef XToWorld(np.ndarray[DTYPE_t, ndim=1] pose):
    # 提前声明类型，便于优化
    cdef DTYPE_t x, y, z, roll, yaw, pitch
    cdef DTYPE_t c_y, s_y, c_r, s_r, c_p, s_p
    cdef np.ndarray[DTYPE_t, ndim=2] matrix

    # 提取pose中的值
    x, y, z, roll, yaw, pitch = pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]

    # 计算三角函数值（使用 numpy）
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    # 初始化4x4单位矩阵
    matrix = np.eye(4, dtype=np.float64)

    # 填充平移矩阵
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # 填充旋转矩阵
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix

def X1ToX2(np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2):
    cdef np.ndarray[DTYPE_t, ndim=2] x1_to_world = XToWorld(x1), x2_to_world = XToWorld(x2)
    cdef np.ndarray[DTYPE_t, ndim=2] world_to_x2 = np.linalg.inv(x2_to_world)
    cdef np.ndarray[DTYPE_t, ndim=2]transformation_matrix = np.dot(world_to_x2, x1_to_world)
    return transformation_matrix