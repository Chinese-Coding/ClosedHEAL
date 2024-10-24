import numpy as np
cimport numpy as cnp

# 定义类型
ctypedef cnp.float64_t F64_t

cdef cnp.ndarray[F64_t, ndim=2] XToWorld(cnp.ndarray[F64_t, ndim=1] pose):
    # 提前声明类型，便于优化
    cdef F64_t x, y, z, roll, yaw, pitch
    cdef F64_t c_y, s_y, c_r, s_r, c_p, s_p

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
    cdef cnp.ndarray[F64_t, ndim=2] matrix = np.eye(4, dtype=np.float64)

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

cpdef cnp.ndarray[F64_t, ndim=2] X1ToX2(cnp.ndarray[F64_t, ndim=1] x1, cnp.ndarray[F64_t, ndim=1] x2):
    cdef cnp.ndarray[F64_t, ndim=2] x1_to_world, x2_to_world, world_to_x2, transformation_matrix

    x1_to_world, x2_to_world = XToWorld(x1), XToWorld(x2)
    world_to_x2 = np.linalg.inv(x2_to_world)
    transformation_matrix = np.dot(world_to_x2, x1_to_world)
    return transformation_matrix

def GetPairwiseTransformation(dict base_data_dict, int max_cav, proj_first):
    """
    获得 `base_data_dict` 中两两车辆之间的坐标转换矩阵

    """
    cdef cnp.ndarray[F64_t, ndim=4] pairwise_t_matrix = np.tile(np.eye(4), (max_cav, max_cav, 1, 1))  # (L, L, 4, 4)
    cdef list[cnp.ndarray[F64_t]] t_list
    cdef int t_list_len

    if proj_first:
        return pairwise_t_matrix
    else:
        t_list = [XToWorld(cav_content.params["lidar_pose"]) for cav_content in base_data_dict.values()]
        t_list_len = len(t_list)

        for i in range(t_list_len):
            for j in range(t_list_len):
                # identity matrix to self
                if i != j:
                    # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                    # t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                    pairwise_t_matrix[i, j] = np.linalg.solve(t_list[j], t_list[i]) # Tjw*Twi = Tji

    return pairwise_t_matrix
