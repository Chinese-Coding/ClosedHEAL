import copy
from typing import Dict

import numpy as np

from opencood.data_utils.data_models.dataset_models import CAVData


def _GenerateNoise(pos_std: float, rot_std: float, pos_mean=0, rot_mean=0, method=np.random.normal):
    """
    将定位误差添加到 6dof 位姿噪声中, 包括位置 (x, y) 和旋转 (yaw).
    :param pos_std:
    :param rot_std:
    :param pos_mean:
    :param rot_mean:
    :param method: 生成噪声的方法 (np.random.normal(正态分布) 或者 np.random.laplace(拉普拉斯分布))
    :return:
    """
    if method not in {np.random.normal, np.random.laplace}:
        raise Exception("不受支持的函数")
    xy, yaw = method(pos_mean, pos_std, size=2), method(rot_mean, rot_std, size=1)
    pose_noise = np.array([xy[0], xy[1], 0, 0, yaw[0], 0])
    return pose_noise


def _AddNoise1(basedata: Dict[str, CAVData]):
    for sensors_data in basedata.values():
        # 这里直接赋值, 也就是说 `lidar_pose` 与 `lidar_pose_clean` 指向的是同一个对象
        sensors_data.params["lidar_pose_clean"] = sensors_data.params["lidar_pose"]
        # 如果说想要不同对象的话, 应该使用 copy.deepcopy 函数, 也就是下面这行代码 (执行下面的测试语句时, is 语句输出 False):
        # agent_content.yaml_data.lidar_pose_clean = copy.deepcopy(agent_content.yaml_data.lidar_pose)

    return basedata


def _AddNoise2(basedata: Dict[str, CAVData], pos_std: float, rot_std: float, pos_mean: int, rot_mean: int, method):
    for sensors_data in basedata.values():
        sensors_data.params["lidar_pose_clean"] = copy.deepcopy(sensors_data.params["lidar_pose"])
        noise = _GenerateNoise(pos_std, rot_std, pos_mean, rot_mean, method)
        sensors_data.params["lidar_pose_clean"] += noise

    return basedata


def GetNoiseGenerator(noiseSetting: Dict):
    """
    创建添加噪声的函数, 根据传入的配置信息, 选择添加噪声的方式.

    * 如果不添加噪声, 则只是单纯的将 `lidar_pose` 赋值 (**没有进行拷贝, 二者指向同一个对象**)给 `lidar_pose_clean`, 并不向其中添加噪声

    * 如果添加噪声, 则根据配置信息, 生成对应产生噪音的函数 (有 `np.random.normal` 和 `np.random.laplace` 两种方法).
      当然在将噪声添加到 `lidar_pose` 之前, 会对 `lidar_pose` 深拷贝后赋值给 `lidar_pose_clean`.

    :param noiseSetting:
    :return:
    """
    if not noiseSetting.get("add_noise", False):  # 默认不添加噪声
        return _AddNoise1

    noiseArgs = noiseSetting.get("args", {})
    laplaceNoise = noiseArgs.get("laplace_noise", False)
    method = np.random.laplace if laplaceNoise else np.random.normal
    pos_std, rot_std, pos_mean, rot_mean = noiseArgs["pos_std"], noiseArgs["rot_std"], noiseArgs["pos_mean"], noiseArgs["rot_mean"]  # fmt: skip

    return lambda basedata: _AddNoise2(basedata, pos_std, rot_std, pos_mean, rot_mean, method)
