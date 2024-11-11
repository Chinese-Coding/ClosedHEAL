import numpy as np
from opencood.utils.cython.all_use cimport cnp, F64_t, I64_t

from opencood.utils.cython.box_utils cimport BoxesToCorners3D, Corner2dToStandupBox




cdef _GetAnchorArgs(dict anchorArgs):
    return (
        anchorArgs["W"], anchorArgs["H"], anchorArgs["l"], anchorArgs["w"], anchorArgs["h"], anchorArgs["r"],
        anchorArgs["vh"], anchorArgs["vw"], anchorArgs["cav_lidar_range"], anchorArgs.get("feature_stride", 2)
    )


cdef _GetCenter(int anchor_num, cnp.ndarray[F64_t, ndim=1] x, cnp.ndarray[F64_t, ndim=1] y):
    cdef cnp.ndarray[F64_t, ndim=2] cx2d, cy2d
    cdef cnp.ndarray[F64_t, ndim=3] cx, cy, cz

    cx2d, cy2d = np.meshgrid(x, y)
    cx, cy = np.tile(cx2d[..., np.newaxis], anchor_num), np.tile(cy2d[..., np.newaxis], anchor_num)
    cz = np.ones_like(cx) * -1.0
    return cx, cy, cz


def GenerateAnchorBox(dict anchorArgs, int anchor_num, str order):
    # load_voxel_params and load_point_pillar_params leads to the same anchor
    # if voxel_size * feature stride is the same.
    cdef int W, H, feature_stride
    cdef double l, w, h, vh, vw
    cdef list[double] r, cav_lidar_range, xrange, yrange

    cdef cnp.ndarray[F64_t, ndim=1] x, y
    cdef cnp.ndarray[F64_t, ndim=3] cx, cy, cz, w_, l_, h_, r_

    W, H, l, w, h, r, vh, vw, cav_lidar_range, feature_stride = _GetAnchorArgs(anchorArgs)
    assert anchor_num == len(r)

    r = [np.radians(ele) for ele in r]

    xrange, yrange = [cav_lidar_range[0], cav_lidar_range[3]], [cav_lidar_range[1], cav_lidar_range[4]]

    # vw is not precise, vw * feature_stride / 2 should be better?
    x, y = np.linspace(xrange[0] + vw, xrange[1] - vw, W // feature_stride), np.linspace(yrange[0] + vh, yrange[1] - vh, H // feature_stride) # fmt: skip
    cx, cy, cz = _GetCenter(anchor_num, x, y)
    w_, l_, h_, r_ = np.full_like(cx, w), np.full_like(cx, l), np.full_like(cx, h), np.full_like(cx, r)

    if order == "hwl":
        return np.stack([cx, cy, cz, h_, w_, l_, r_], axis=-1)
    elif order == "lhw":
        return np.stack([cx, cy, cz, l_, h_, w_, r_], axis=-1)
    else:
        raise NotImplementedError(f"{order} is unknown bbx order.")


cdef cnp.ndarray[F64_t, ndim=2] _BboxOverlaps(cnp.ndarray[F64_t, ndim=2] boxes, cnp.ndarray[F64_t, ndim=2] query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef cnp.ndarray[F64_t, ndim=2] overlaps = np.zeros((N, K))
    cdef double iw, ih, box_area
    cdef double ua
    cdef unsigned int k, n
    for k in range(K):
        box_area = (query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1)

        for n in range(N):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1

            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1

                if ih > 0:
                    ua = float((boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1) + box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def GenerateLabel(
        cnp.ndarray[F64_t, ndim=2] gt_box_center, cnp.ndarray[F64_t, ndim=4] anchors, cnp.ndarray[I64_t, ndim=1] masks,
        int anchor_num, str order, double targetPosThreshold, double targetNegThreshold,
) -> dict[str, cnp.ndarray[F64_t]]:
    """
    `max_num` 是配置文件中的参数; `H`, `W` 指的是特征图的大小 (具体也是由配置文件里面的参数决定的, 但是应该不能直接看到)
    `anchor_num` 也是配置文件里面的参数; `7` 一种坐标表示形式
    :param gt_box_center: shape: (max_num, 7)
    :param anchors:       shape: (H, W, anchor_num, 7)
    :param masks:         shape: (max_num, )
    :param anchor_num: 单个点生成的 anchor 个数, 一般是 2
    """
    # `cdef tuple[unsigned int, unsigned int] ` 去掉，不兼容
    feature_map_shape = (anchors.shape[0], anchors.shape[1]) # shape: (H, W)

    cdef cnp.ndarray[F64_t, ndim=2] anchors2D = anchors.reshape(-1, 7)  # shape: (H * W * anchor_num, 7)
    cdef cnp.ndarray[F64_t, ndim=1] anchors1D = np.sqrt(anchors2D[:, 4] ** 2 + anchors2D[:, 5] ** 2) # shape: (H * W * anchor_num)

    cdef cnp.ndarray[F64_t, ndim=3] pos_equal_one, neg_equal_one, targets

    pos_equal_one, neg_equal_one = np.zeros((*feature_map_shape, anchor_num)), np.zeros((*feature_map_shape, anchor_num)) # shape:  (H, W, anchor_num)
    targets = np.zeros((*feature_map_shape, anchor_num * 7)) # shape: (H, W, anchor_num * 7)

    cdef cnp.ndarray[F64_t, ndim=2] gt_box_center_valid, anchors_standup_2d, gt_standup_2d
    cdef cnp.ndarray[F64_t, ndim=3] gt_box_center_valid3D, anchors_corner

    gt_box_center_valid = gt_box_center[masks == 1]                              # (n, 7)
    gt_box_center_valid3D = BoxesToCorners3D(gt_box_center_valid, order) # (n, 8, 3)
    anchors_corner = BoxesToCorners3D(anchors2D, order)                  # (H * W * anchor_num, 8, 3)
    gt_standup_2d = Corner2dToStandupBox(gt_box_center_valid3D)       # (n, 4)
    anchors_standup_2d = Corner2dToStandupBox(anchors_corner)       # (H * W * anchor_num, 4)

    cdef cnp.ndarray[I64_t, ndim=1] id_highest, id_highest_gt, id_pos, id_pos_gt, index, id_neg
    cdef cnp.ndarray[F64_t, ndim=2] iou

    # (H*W*anchor_n)
    iou = _BboxOverlaps(np.ascontiguousarray(anchors_standup_2d), np.ascontiguousarray(gt_standup_2d))

    # the anchor boxes has the largest iou across
    # shape: (n)
    id_highest = np.argmax(iou.T, axis=1)
    # [0, 1, 2, ..., n-1]
    id_highest_gt = np.arange(iou.T.shape[0])
    # make sure all highest iou is larger than 0
    mask = iou.T[id_highest_gt, id_highest] > 0
    id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

    # find anchors iou > params['pos_iou']
    id_pos, id_pos_gt = np.where(iou > targetPosThreshold)
    #  find anchors iou < params['neg_iou']
    id_neg = np.where(np.sum(iou < targetNegThreshold, axis=1) == iou.shape[1])[0]
    id_pos, id_pos_gt = np.concatenate([id_pos, id_highest]), np.concatenate([id_pos_gt, id_highest_gt])
    id_pos, index = np.unique(id_pos, return_index=True)
    id_pos_gt = id_pos_gt[index]
    id_neg.sort()

    cdef cnp.ndarray[I64_t, ndim=1] index_x, index_y, index_z,

    # cal the target and set the equal one
    index_x, index_y, index_z = np.unravel_index(id_pos, (*feature_map_shape, anchor_num))
    pos_equal_one[index_x, index_y, index_z] = 1

    # calculate the targets
    targets[index_x, index_y, np.array(index_z) * 7] = (gt_box_center[id_pos_gt, 0] - anchors2D[id_pos, 0]) / anchors1D[id_pos]
    targets[index_x, index_y, np.array(index_z) * 7 + 1] = (gt_box_center[id_pos_gt, 1] - anchors2D[id_pos, 1]) / anchors1D[id_pos]
    targets[index_x, index_y, np.array(index_z) * 7 + 2] = (gt_box_center[id_pos_gt, 2] - anchors2D[id_pos, 2]) / anchors2D[id_pos, 3]
    targets[index_x, index_y, np.array(index_z) * 7 + 3] = np.log(gt_box_center[id_pos_gt, 3] / anchors2D[id_pos, 3])
    targets[index_x, index_y, np.array(index_z) * 7 + 4] = np.log(gt_box_center[id_pos_gt, 4] / anchors2D[id_pos, 4])
    targets[index_x, index_y, np.array(index_z) * 7 + 5] = np.log(gt_box_center[id_pos_gt, 5] / anchors2D[id_pos, 5])
    targets[index_x, index_y, np.array(index_z) * 7 + 6] = gt_box_center[id_pos_gt, 6] - anchors2D[id_pos, 6]

    index_x, index_y, index_z = np.unravel_index(id_neg, (*feature_map_shape, anchor_num))
    neg_equal_one[index_x, index_y, index_z] = 1

    # to avoid a box be pos/neg in the same time
    index_x, index_y, index_z = np.unravel_index(id_highest, (*feature_map_shape, anchor_num))
    neg_equal_one[index_x, index_y, index_z] = 0

    return {"pos_equal_one": pos_equal_one, "neg_equal_one": neg_equal_one, "targets": targets}
