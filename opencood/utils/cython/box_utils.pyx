import numpy as np

from opencood.utils.cython.transformation_utils cimport X1ToX2


def ProjectPointsByMatrixFloat32(
        cnp.ndarray[F32_t, ndim=2] points,
        cnp.ndarray[F32_t, ndim=2] transformation_matrix
):
    cdef cnp.ndarray[F32_t, ndim=2] points_homogeneous, projected_points

    # convert to homogeneous coordinates via padding 1 at the last dimension
    points_homogeneous = np.pad(points, ((0, 0), (0, 1)), mode="constant", constant_values=1)
    # 使用 numpy 的高效矩阵乘法
    projected_points = np.dot(points_homogeneous, transformation_matrix.T)

    return projected_points[:, :3]

def ProjectPointsByMatrixFloat64(
        cnp.ndarray[F64_t, ndim=2] points,
        cnp.ndarray[F64_t, ndim=2] transformation_matrix
):
    cdef cnp.ndarray[F64_t, ndim=2] points_homogeneous, projected_points

    # convert to homogeneous coordinates via padding 1 at the last dimension
    points_homogeneous = np.pad(points, ((0, 0), (0, 1)), mode="constant", constant_values=1)
    # 使用 numpy 的高效矩阵乘法
    projected_points = np.dot(points_homogeneous, transformation_matrix.T)

    return projected_points[:, :3]

cdef cnp.ndarray[F64_t, ndim=2] CreateBbx(list[double] extent):
    """
    Create bounding box with 8 corners under obstacle vehicle reference.
    """

    return np.array([
        [extent[0], -extent[1], -extent[2]],
        [extent[0], extent[1], -extent[2]],
        [-extent[0], extent[1], -extent[2]],
        [-extent[0], -extent[1], -extent[2]],
        [extent[0], -extent[1], extent[2]],
        [extent[0], extent[1], extent[2]],
        [-extent[0], extent[1], extent[2]],
        [-extent[0], -extent[1], extent[2]],
    ])

cdef cnp.ndarray[F64_t, ndim=2] CornerToCenter(cnp.ndarray[F64_t, ndim=3] corner3d, str order="lwh"):
    """
    Convert 8 corners to x, y, z, dx, dy, dz, yaw.
    yaw in radians

    :param corner3d: shape: (N, 8, 3)
    :param order:

    box3d : np.ndarray
        (N, 7)
    """
    cdef int batch_size = corner3d.shape[0]

    cdef cnp.ndarray[F64_t, ndim=2] xyz, h, l, w, theta

    xyz = np.mean(corner3d[:, [0, 3, 5, 6], :], axis=1)

    h = abs(np.mean(corner3d[:, 4:, 2] - corner3d[:, :4, 2], axis=1, keepdims=True))

    l = (
        np.sqrt(np.sum((corner3d[:, 0, [0, 1]] - corner3d[:, 3, [0, 1]]) ** 2, axis=1, keepdims=True))
        + np.sqrt(np.sum((corner3d[:, 2, [0, 1]] - corner3d[:, 1, [0, 1]]) ** 2, axis=1, keepdims=True))
        + np.sqrt(np.sum((corner3d[:, 4, [0, 1]] - corner3d[:, 7, [0, 1]]) ** 2, axis=1, keepdims=True))
        + np.sqrt(np.sum((corner3d[:, 5, [0, 1]] - corner3d[:, 6, [0, 1]]) ** 2, axis=1, keepdims=True))
    ) / 4

    w = (
        np.sqrt(np.sum((corner3d[:, 0, [0, 1]] - corner3d[:, 1, [0, 1]]) ** 2, axis=1, keepdims=True))
        + np.sqrt(np.sum((corner3d[:, 2, [0, 1]] - corner3d[:, 3, [0, 1]]) ** 2, axis=1, keepdims=True))
        + np.sqrt(np.sum((corner3d[:, 4, [0, 1]] - corner3d[:, 5, [0, 1]]) ** 2, axis=1, keepdims=True))
        + np.sqrt(np.sum((corner3d[:, 6, [0, 1]] - corner3d[:, 7, [0, 1]]) ** 2, axis=1, keepdims=True))
    ) / 4

    theta = (
        np.arctan2(corner3d[:, 1, 1] - corner3d[:, 2, 1], corner3d[:, 1, 0] - corner3d[:, 2, 0])
        + np.arctan2(corner3d[:, 0, 1] - corner3d[:, 3, 1], corner3d[:, 0, 0] - corner3d[:, 3, 0])
        + np.arctan2(corner3d[:, 5, 1] - corner3d[:, 6, 1], corner3d[:, 5, 0] - corner3d[:, 6, 0])
        + np.arctan2(corner3d[:, 4, 1] - corner3d[:, 7, 1], corner3d[:, 4, 0] - corner3d[:, 7, 0])
    )[:, np.newaxis] / 4

    if order == "lwh":
        return np.concatenate([xyz, l, w, h, theta], axis=1).reshape(batch_size, 7)
    elif order == "hwl":
        return np.concatenate([xyz, h, w, l, theta], axis=1).reshape(batch_size, 7)
    else:
        raise ValueError("不受支持的 order 类型")



cpdef cnp.ndarray[F64_t, ndim=3] BoxesToCorners3D(cnp.ndarray[F64_t, ndim=2] boxes3d, str order):
    """
    将三维边界框参数转换为其对应的八个角点坐标。

    参数
    ----------
    boxes3d : np.ndarray
        形状为 (N, 7) 的数组，表示 N 个三维边界框。
        每个边界框的参数可以是 [x, y, z, l, w, h, heading]，
        或者 [x, y, z, h, w, l, heading]，取决于 `order` 参数。
        - (x, y, z) 是边界框的中心点坐标。
        - l, w, h 分别是边界框的长度、宽度和高度。
        - heading 是边界框绕 z 轴的旋转角度（弧度）。

    order : str
        尺寸顺序，可以是 'lwh' 或 'hwl'。默认为 'lwh'。

    返回
    -------
    corners3d : np.ndarray
        形状为 (N, 8, 3) 的数组，表示每个边界框的八个角点的三维坐标。

    示例
    -------
    ```python
    boxes3d = np.array([
        [0, 0, 0, 2, 4, 6, np.pi / 4]  # [x, y, z, l, w, h, heading]
    ])
    corners = boxes_to_corners_3d_numpy(boxes3d, order='lwh')
    print(corners)
    ```
    """


    if order == "hwl": # 处理尺寸顺序
        boxes3d = boxes3d[:, [0, 1, 2, 5, 4, 3, 6]]

    cdef cnp.ndarray[F64_t, ndim=1] headings, cos_h, sin_h
    cdef cnp.ndarray[F64_t, ndim=2] centers, dims
    cdef cnp.ndarray[F64_t, ndim=3] corners, rotation_matrices, corners_rotated
    cdef cnp.ndarray[F64_t, ndim=2] template

    # 提取中心点、尺寸和旋转角度
    centers, dims = boxes3d[:, 0:3], boxes3d[:, 3:6]
    headings = boxes3d[:, 6]

    # 定义八个角点的相对位置（单位立方体）
    template = np.array([
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, -1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
        [-1, -1, 1],
    ]) / 2

    # 扩展维度并缩放到边界框尺寸
    corners = dims[:, np.newaxis, :] * template[np.newaxis, :, :]  # 广播相乘

    # 计算绕 z 轴的旋转矩阵
    cos_h, sin_h = np.cos(headings), np.sin(headings)

    rotation_matrices = np.zeros((boxes3d.shape[0], 3, 3))
    rotation_matrices[:, 0, 0] = cos_h
    rotation_matrices[:, 0, 1] = -sin_h
    rotation_matrices[:, 1, 0] = sin_h
    rotation_matrices[:, 1, 1] = cos_h
    rotation_matrices[:, 2, 2] = 1

    # 旋转角点, 使用 einsum 进行批量矩阵乘法
    corners_rotated = np.einsum('nij,nkj->nki', rotation_matrices, corners)
    # 平移到中心点位置
    corners_rotated += centers[:, np.newaxis, :]

    return corners_rotated

cdef cnp.ndarray[F64_t, ndim=3] MaskBoxesOutsideRangeWith7False(cnp.ndarray[F64_t, ndim=2] boxes, cnp.ndarray[F64_t, ndim=1] limit_range, str order, int mini_num_corners=8):
    """
    函数 `MaskBoxesOutsideRange` 的特殊形式, `With7False` 表示 `boxes.shape[1] == 7` 并且原函数中的 `return_mask=False`
    """
    cdef cnp.ndarray[F64_t, ndim=2] new_boxes = boxes.copy()
    cdef cnp.ndarray[cnp.npy_bool, ndim=2] mask
    cdef cnp.ndarray[F64_t, ndim=3] boxes3D = BoxesToCorners3D(new_boxes, order)

    mask = ((boxes3D >= limit_range[0:3]) & (boxes3D <= limit_range[3:6])).all(axis=2)
    return  boxes[mask.sum(axis=1) >= mini_num_corners]


cpdef dict ProjectWorldObjects(dict[str, dict] object_dict, cnp.ndarray[F64_t, ndim=1] lidar_pose, cnp.ndarray[F64_t, ndim=1] lidar_range, str order, enlarge_z=False):
    if enlarge_z:
        lidar_range = lidar_range[:]  # 浅拷贝一下
        lidar_range[2], lidar_range[5] = lidar_range[2] - 10, lidar_range[5] + 10

    output_dict = {}
    cdef cnp.ndarray[F64_t, ndim=2] object2lidar, bbx, bbx_lidar
    cdef list[double] location, rotation, center, extent
    cdef cnp.ndarray[F64_t, ndim=1] object_pose

    for object_id, content in object_dict.items():

        location, rotation, center, extent = (
            content["location"], content["angle"], content.get("center", [0, 0, 0]), content["extent"],
        )
        # 计算物体的姿态
        object_pose = np.array([
            location[0] + center[0], location[1] + center[1], location[2] + center[2],
            rotation[0], rotation[1], rotation[2],
        ])

        object2lidar = X1ToX2(object_pose, lidar_pose)  # 物体姿态转换到激光雷达坐标系
        bbx = np.vstack((CreateBbx(extent).T, np.ones((1, 8))))  # 创建物体的 bbx，shape (4, 8)，添加一行全 1 用于坐标变换
        bbx_lidar = np.dot(object2lidar, bbx).T[:, :3]  # 只保留前三列 (x, y, z) # 将 bounding box 投影到世界坐标系
        bbx_lidar = CornerToCenter(np.expand_dims(bbx_lidar, 0), order=order)  # 将角点转换为中心表示
        bbx_lidar = MaskBoxesOutsideRangeWith7False(bbx_lidar, lidar_range, order)  # 根据范围过滤

        if bbx_lidar.shape[0] > 0:  # 如果过滤后仍有 box，更新到输出字典
            output_dict[object_id] = bbx_lidar
    return output_dict

cpdef cnp.ndarray[F64_t, ndim=2] Corner2dToStandupBox(cnp.ndarray[F64_t, ndim=3] box2d):
    """
    Find the minmaxx, minmaxy for each 2d box. (N, 4, 2) -> (N, 4)
    x1, y1, x2, y2

    Parameters
    ----------
    box2d : np.ndarray
        (n, 4, 2), four corners of the 2d bounding box.

    Returns
    -------
    standup_box2d : np.ndarray
        (n, 4)
    """
    cdef int N = box2d.shape[0]
    cdef cnp.ndarray[F64_t, ndim=2] standup_boxes2d = np.zeros((N, 4))

    standup_boxes2d[:, 0] = np.min(box2d[:, :, 0], axis=1)
    standup_boxes2d[:, 1] = np.min(box2d[:, :, 1], axis=1)
    standup_boxes2d[:, 2] = np.max(box2d[:, :, 0], axis=1)
    standup_boxes2d[:, 3] = np.max(box2d[:, :, 1], axis=1)

    return standup_boxes2d


