import os
import random
from typing import List, Dict

import numpy as np

from opencood.utils.file_utils import load_json
from opencood.utils.globals import MODEL_MODE


class Adaptor:
    """适用于异质编程的转换器 (TODO: 没有完全抄过来, 只是抄了 IntermHeter 中使用到的函数和变量)"""

    def __init__(self, configs: Dict):
        config = configs.get("heter", configs)  # 兼容两种传参模式 (configs["heter"] 或 configs)

        self.ego_modality: List[str] = config["ego_modality"].split("&")

        assignment_path = config.get("assignment_path", None)
        self.modality_assignment = None if assignment_path is None else load_json(os.path.expanduser(assignment_path))
        self.mapping_rules = config["mapping_dict"]
        self.model_modalities = list(config["modality_setting"].keys())
        # 获取各个模态的权重, 如果没有指定, 则进行平均分配. 这里由原先的字典, 变成了和 `model_modalities` 一一对应的列表
        length = len(self.model_modalities)
        # TODO: 单纯根据复现的配置文件来看, 配置文件中并没有这一项, 原项目这么写估计是为了方便后续拓展
        self.model_modality_weights = config.get("cav_preference", [1 / length for _ in range(length)])
        self.lidar_channels_dict = config.get("lidar_channels_dict", {})

    def unmatched_modality(self, agent_modality):
        """
        检查智能体的模态是否在模型允许的模态中
        :param agent_modality:
        :return: 如果不在返回 `True`, 如果在返回 `False`
        """
        return agent_modality not in self.model_modalities

    def assign_modality(self, modality_name: str, ego: bool):
        """
        返回 `modality_name` 对应的模态.

        如果不在训练模式则返回 `mapping_dict` 中的模态.

        如果处于训练模式: 如果是 ego 则从 `ego_modality` 中随机进行选择; 如果不是, 则根据不同模态的权重随机选择一个
        :param modality_name: 模态名称
        :param ego: 是否为自智能体
        :return: 分配的模态
        """
        if not MODEL_MODE:
            return self.mapping_rules[modality_name]
        if ego:
            return np.random.choice(self.ego_modality)
        return random.choices(self.model_modalities, weights=self.model_modality_weights)[0]

    def switch_lidar_channels(self, agent_modality: str, lidar_file: str):
        """Currently only support OPV2V"""
        lidar_channel = self.lidar_channels_dict.get(agent_modality)
        if lidar_channel in {16, 32}:
            # WARNING: 直接将替换后的文件路径写到代码里面是否合适呢?
            return lidar_file.replace("OPV2V", "OPV2V-H").replace(".pcd", f"_{lidar_channel}.pcd")
        return lidar_file

    def reorder_agents(self, agent_ids: List[str], scenario_name: str):
        """
        评估时, 将映射后可能为自我模态的cav作为第一个，这样可以检查对齐器的训练效果。
        # TODO: 这个函数还需要进一步的修改
        :param agent_ids:
        :param scenario_name:
        :return:
        """
        scenario_modality_assignment = self.modality_assignment[scenario_name]
        if scenario_modality_assignment[agent_ids[0]] not in self.model_modalities:
            ego_agent_id = None
            for agent_id, modality in scenario_modality_assignment:
                if self.mapping_rules[modality] in self.model_modalities:
                    ego_agent_id = agent_id
                    break
            if ego_agent_id is None:
                return agent_ids

            other_agent_ids = sorted(list(scenario_modality_assignment.keys()))
            other_agent_ids.remove(ego_agent_id)
            agent_ids = [ego_agent_id] + other_agent_ids
        return agent_ids
