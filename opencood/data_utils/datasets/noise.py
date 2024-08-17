import copy
from typing import Dict
import numpy as np


from opencood.data_utils.datasets.dataset_models import SensorsData


def _generate_noise(pos_std: float, rot_std: float, pos_mean=0, rot_mean=0, method=np.random.normal):
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


def _add_noise_1(basedata: Dict[str, SensorsData]):
    for agent_id, agent_content in basedata.items():
        # 这里直接赋值, 也就是说 `lidar_pose` 与 `lidar_pose_clean` 指向的是同一个对象
        agent_content.yaml_data.lidar_pose_clean = agent_content.yaml_data.lidar_pose
        # 如果说想要不同对象的话, 应该使用 copy.deepcopy 函数, 也就是下面这行代码 (执行下面的测试语句时, is 语句输出 False):
        # agent_content.yaml_data.lidar_pose_clean = copy.deepcopy(agent_content.yaml_data.lidar_pose)

    return basedata


def _add_noise_2(basedata: Dict[str, SensorsData], pos_std: float, rot_std: float, pos_mean: int, rot_mean: int, method):
    for agent_id, agent_content in basedata.items():
        agent_content.yaml_data.lidar_pose_clean = copy.deepcopy(agent_content.yaml_data.lidar_pose)
        noise = _generate_noise(pos_std, rot_std, pos_mean, rot_mean, method)
        agent_content.yaml_data.lidar_pose += noise

    return basedata


def create_noise_generator(noise_setting: Dict):
    """
    创建添加噪声的函数, 根据传入的配置信息, 选择添加噪声的方式.

    * 如果不添加噪声, 则只是单纯的将 `lidar_pose` 赋值 (**没有进行拷贝, 二者指向同一个对象**)给 `lidar_pose_clean`, 并不向其中添加噪声

    * 如果添加噪声, 则根据配置信息, 生成对应产生噪音的函数 (有 `np.random.normal` 和 `np.random.laplace` 两种方法).
      当然在将噪声添加到 `lidar_pose` 之前, 会对 `lidar_pose` 深拷贝后赋值给 `lidar_pose_clean`.

    **一般而言, 只使用到了不添加噪声的情况, 所以没有对添加噪声的部分进行测试.**
    :param noise_setting:
    :return:
    """
    if not noise_setting.get("add_noise", False):  # 默认不添加噪声
        return _add_noise_1

    noise_args = noise_setting.get("args", {})
    laplace_noise = noise_args.get("laplace_noise", False)
    method = np.random.laplace if laplace_noise else np.random.normal
    pos_std, rot_std, pos_mean, rot_mean = noise_args['pos_std'], noise_args['rot_std'], noise_args['pos_mean'], noise_args['rot_mean']  # fmt: skip

    return lambda basedata: _add_noise_2(basedata, pos_std, rot_std, pos_mean, rot_mean, method)


if __name__ == "__main__":
    from icecream import ic
    import os
    from opencood.utils.yaml_utils import load_yaml
    from opencood.data_utils.datasets.base_datasets.opv2v_dataset import OPV2VDataset

    config_path = "~/PycharmProjects/ClosedHEAL/opencood/configs/OPV2V/MoreModality/HEAL/stage1/m1_pyramid.yaml"
    config_path = os.path.expanduser(config_path)
    configs = load_yaml(config_path)
    opv2v_dataset = OPV2VDataset(configs, False)
    opv2v_dataset.initialize()
    basedata = opv2v_dataset[0]
    add_noise = create_noise_generator(configs.get("noise_setting", {}))
    basedata = add_noise(basedata)
    for agent_id, agent_content in basedata.items():
        lidar_pose, lidar_pose_clean = agent_content.yaml_data.lidar_pose, agent_content.yaml_data.lidar_pose_clean
        ic(lidar_pose is lidar_pose_clean)
        ic(lidar_pose == lidar_pose_clean)
