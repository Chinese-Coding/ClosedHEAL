from dataclasses import dataclass
from typing import List, NoReturn, Optional, Dict

import numpy as np


@dataclass
class Timestamp:
    timestamp: str
    """以下三个字段包含着对应数据文件的路径, 并不是真的文件"""
    yaml: str
    lidar: str
    cameras: List[str]
    depths: List[str]
    agent_modality: Optional[str] = None  # 异质的是否使用


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
class Scenario:
    name: str
    agents: List[Agent]

    def __init__(self, name: str, agents=None):
        self.name = name
        self.agents = [] if agents is None else agents

    def add_agent(self, agent: Agent) -> NoReturn:
        self.agents.append(agent)


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
class YamlData:
    cameras: List[Camera]
    ego_speed: float
    lidar_pose: List[float]
    plan_trajectory: List[List[float]]
    predicted_ego_pos: List[float]
    true_ego_pos: List[float]
    vehicles: Dict[int, Vehicle]
    lidar_pose_clean: Optional[List[float]] = None


@dataclass
class RetrievedData:
    """
    经数据集 retrieve 后返回的数据
    主要是某一 timestamp 下读取的真实数据, 也包括一些额外数据.
    """
    id: str
    ego: bool
    yaml_data: YamlData  # 就是原先的 'params' 字段, 里面的东西都是一层层的字典很复杂
    lidar_data: Optional[np.ndarray]  # 从后缀为 .pcd 的文件读取
    camera_data: Optional[List[Dict]]
    depth_data: Optional[List[Dict]]
    agent_modality: Optional[str] = None  # agent 的模态, 非异质的模型不需要i这个参数
