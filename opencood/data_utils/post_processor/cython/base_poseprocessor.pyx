from collections import ChainMap

import numpy as np
cimport numpy as cnp
cimport cython

ctypedef cnp.float64_t F64_t
ctypedef list List
ctypedef dict Dict

from opencood.utils.cython cimport box_utils

class BasePostprocessor:
    cdef str order
    cdef int maxNUm
    cdef bool train
    cdef np.ndarray[F64_t, ndim=1] filter_range_numpy

    def __init__(self, dict anchor_params, train=True):
        self.train = train
        self.order, self.maxNUm = anchor_params["order"], anchor_params["max_num"]
        self.filter_range_numpy = np.array(anchor_params["anchor_args"]["cav_lidar_range"] if self.train else anchor_params["gt_range"])

    def GenerateObjectCenter(self, list cav_contents, cnp.ndarray[F64_t] reference_lidar_pose, enlarge_z=False):
        # 使用 ChainMap 合并多个字典，避免重复创建中间字典
        tmp_object_dict: Dict[int, np.ndarray] = dict(ChainMap(*(cav_content.params["vehicles"] for cav_content in cav_contents))) # fmt: skip

        output_dict: Dict[int, np.ndarray[np.float64]] = box_utils.ProjectWorldObjects(
            tmp_object_dict, reference_lidar_pose, self.filter_range_numpy, self.order, enlarge_z
        )
        numObjs = min(len(output_dict), self.maxNum)

        object_np, mask, object_ids = np.zeros((self.maxNum, 7)), np.zeros(self.maxNum), list(output_dict.keys())[:numObjs]
        if numObjs != 0:  # 需要对 `0` 这种情况做一个特殊处理
            object_np[:numObjs], mask[:numObjs] = np.array([output_dict[objId][0, :] for objId in object_ids]), np.ones(numObjs)

        return object_np, mask.astype(np.int64), object_ids