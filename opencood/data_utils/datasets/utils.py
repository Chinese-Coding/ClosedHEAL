from typing import List

import numpy as np
import open3d
from PIL import Image

from opencood.data_utils.datasets.dataset_models import AgentInfo, Camera, Vehicle
from opencood.utils.file_utils import load_yaml


def load_camera_data(camera_files: List[str], preload=True) -> List[Image]:
    """
    加载图像数据
    :param camera_files: store camera path
    :param preload: 是否预先加载 (默认为 True)
    :return: list of Image, RGB order
    """
    camera_data_list = []
    for camera_file in camera_files:
        camera_data = Image.open(camera_file)
        if preload:
            camera_data = camera_data.copy()
        camera_data_list.append(camera_data)
    return camera_data_list


def load_depth_data(depth_files: List[str], preload=True) -> List[Image]:
    """加载深度数据, 可以与加载图像数据共用一个函数"""
    return load_camera_data(depth_files, preload)


def load_lidar_data(lidar_file: str) -> np.array:
    """
    加载 lidar 数据 Read pcd and return numpy array.
    :param lidar_file: The pcd file that contains the point cloud.
    :return: The lidar data in numpy format, shape:(n, 4)
    """
    pcd = open3d.io.read_point_cloud(lidar_file)
    # 注意：xyz 和 intensity 标注的 shape 在每次读取时点的数量可能不同，但是维度是唯一的
    xyz = np.asarray(pcd.points)  # xyz.shape = (58547, 3)
    # we save the intensity (强度) in the first channel
    intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)  # intensity.shape = (58547, 1)
    pcd_np = np.hstack((xyz, intensity))

    return np.asarray(pcd_np, dtype=np.float32)


def load_yaml_data(yaml_file: str) -> AgentInfo:
    """
    Load yaml data from yaml file.
    :param yaml_file: The yaml file that contains the point cloud.
    :return: The lidar data in numpy format, shape:(n, 4)
    """
    data = load_yaml(yaml_file)

    """创建 Camera 列表"""
    cameras = []
    for i in range(4):
        camera = Camera(**data[f"camera{i}"])
        cameras.append(camera)

    """创建 Vehicle 的 Dict"""
    vehicles_dict = data.get("vehicles", {})
    vehicles = {}
    for k, v in vehicles_dict.items():
        vehicles[k] = Vehicle(**v)

    # 将解析的数据封装到YamlData数据类中
    # fmt: off
    return AgentInfo(
        cameras=cameras, ego_speed=data.get('ego_speed', 0.0),
        lidar_pose=data.get('lidar_pose', []), plan_trajectory=data.get('plan_trajectory', []),
        predicted_ego_pos=data.get('predicted_ego_pos', []), true_ego_pos=data.get('true_ego_pos', []),
        vehicles=vehicles
    )
    # fmt


if __name__ == "__main__":
    import os
    from icecream import ic

    yaml_file = "../../../dataset/OPV2V/train/2021_08_16_22_26_54/641/000069.yaml"
    root_path = os.getcwd()
    ic(root_path)
    yaml_data = load_yaml_data(yaml_file)
    ic(yaml_data)
