from dataclasses import dataclass
from typing import List, NoReturn, Optional, Dict

import numpy as np


@dataclass
class Timestamp:
    """以下三个字段包含着对应数据文件的路径, 并不是真的文件"""

    yaml: str
    lidar: str
    cameras: List[str]
    depths: List[str]
    modality: Optional[str] = None  # 异质的时候使用


@dataclass
class Agent:
    id: str
    ego: bool
    timestamps: List[Timestamp]

    def __init__(self, id: str, ego=False, timestamp_datas=None):
        self.id = id
        self.ego = ego
        self.timestamps = [] if timestamp_datas is None else timestamp_datas

    def add_timestamp_data(self, timestamp: Timestamp) -> NoReturn:
        self.timestamps.append(timestamp)


@dataclass
class Camera:
    cords: List[float]
    extrinsic: List[List[float]]
    intrinsic: List[List[float]]


@dataclass
class Vehicle:
    angle: List[float]
    extent: List[float]
    location: List[float]
    speed: float
    center: Optional[List[float]] = None


@dataclass
class AgentInfo:
    """某一时间戳下 yaml 文件中的信息, 存储着此时间戳下的格式车辆信息"""

    cameras: List[Camera]
    ego_speed: float
    lidar_pose: List[float]
    plan_trajectory: List[List[float]]
    predicted_ego_pos: List[float]
    true_ego_pos: List[float]
    vehicles: Dict[int, Vehicle]
    lidar_pose_clean: Optional[List[float]] = None


@dataclass
class SensorsData:
    """经数据集 retrieve 后返回的数据. 主要是某一 timestamp 下读取的真实数据, 也包括一些额外数据."""

    ego: bool
    modality: Optional[str]  # agent 的模态, 非异质的模型不需要这个参数
    yaml_data: AgentInfo  # 就是原先的 'params' 字段, 里面的东西都是一层层的字典很复杂
    lidar_data: Optional[np.ndarray]  # 从后缀为 .pcd 的文件读取
    camera_data: Optional[List[Dict]]
    depth_data: Optional[List[Dict]]
