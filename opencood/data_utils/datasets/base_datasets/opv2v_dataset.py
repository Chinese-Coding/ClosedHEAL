# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os
import random
from pathlib import Path
from typing import List, Dict, Union

import cv2
import numpy as np
from torch.utils.data import Dataset

import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.data_utils.data_models.dataset_models import PFTimestampData, CAVData
from opencood.data_utils.other.noise import GetNoiseGenerator
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.camera_utils import load_camera_data
from opencood.utils.logger import get_logger
from opencood.utils.transformation_utils import x1_to_x2

logger = get_logger()


def _GetTimestampDataPath(cavPath: Path, timestamp: str):
    """
    获取某一时间戳下数据的路径

    :param cavPath 汽车所在路径
    :param timestamp 时间戳
    """
    yaml_file, lidar_file = os.path.join(cavPath, f"{timestamp}.yaml"), os.path.join(cavPath, f"{timestamp}.pcd")
    camera_files, depth_files = [cavPath / f"{timestamp}_camera{i}.png" for i in range(4)], [cavPath / f"{timestamp}_depth{i}.png" for i in range(4)] # fmt: skip
    # 替换 "OPV2V" 为 "OPV2V_Hetero" 在 depth 文件路径中
    depth_files = [p.with_name(p.name.replace("OPV2V", "OPV2V_Hetero")) for p in depth_files]

    return yaml_file, lidar_file, camera_files, depth_files


def _LoadParams(yamlFile: str):
    """
    Load params from YAML (同时将嵌套字典中的列表数据递归转换为 np.ndarray)
    """
    params = load_yaml(yamlFile)

    def _ConvertToArray(data):
        match data:
            case dict():
                return {k: _ConvertToArray(v) for k, v in data.items()}
            case list():
                return np.array(data)
            case _:
                return data

    return _ConvertToArray(params)  # 对 params 进行递归处理


def _ReplaceWithAdditional(filePath: str):
    """Replace the main folder with 'additional' if file is not found."""
    return Path(filePath).with_name(
        filePath.replace("train", "additional/train")
        .replace("validate", "additional/validate")
        .replace("test", "additional/test")
    )


class OPV2VDataset(Dataset):
    def __init__(self, params, visualize, train=True, hetero=False, adaptor=None):
        super().__init__()
        self.visualize, self.train = visualize, train
        if hetero and adaptor is None:
            raise ValueError("adaptor cannot be None when heterogeneous is True")
        self.hetero, self.adaptor = hetero, adaptor

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)

        self.data_augmentor = DataAugmentor(params["data_augment"], train) if "data_augment" in params else None

        root_dir = params["root_dir"] if self.train else params["validate_dir"]

        logger.important(f"从 {root_dir} 中加载数据")

        # 来自GPT: 冗余的默认值处理
        self.max_cav = params.get("train_params", {}).get("max_cav", 5)

        # 来自 GPT, 条件判断的简化
        self.load_lidar_file = "lidar" in params["input_source"] or self.visualize
        self.load_camera_file = "camera" in params["input_source"]
        self.load_depth_file = "depth" in params["input_source"]

        self.label_type = params["label_type"]  # 'lidar' or 'camera'
        self.GenerateObjectCenter = (
            self.GenerateObjectCenterLidar if self.label_type == "lidar" else self.generate_object_center_camera
        )
        self.generate_object_center_single = (
            self.GenerateObjectCenter
        )  # will it follows 'self.generate_object_center' when 'self.generate_object_center' change?

        if self.load_camera_file:
            self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        # by default, we load lidar, camera and metadata. But users may
        # define additional inputs/tasks
        self.add_data_extension = params["add_data_extension"] if "add_data_extension" in params else []

        self.addNoise = GetNoiseGenerator(params.get("noise_setting", {}))

        # first load all paths of different scenarios
        # 来自GPT: 路径处理的优化
        self.scenario_folders: List[Path] = sorted(folder for folder in Path(root_dir).iterdir() if folder.is_dir())

        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database: List[Dict[str, Dict[str, Union[PFTimestampData, bool]]]] = []
        self.len_record = []

    @property
    def MaxCAV(self):
        return self.max_cav

    def reinitialize(self):
        # 每次初始化的时候记得清空之前存储的东西 (如果是第一次初始化可能不需要, 但是为了统一写法就不做判断了)
        self.scenario_database.clear()
        self.len_record.clear()

        # loop over all scenarios
        for i, scenario_folder in enumerate(self.scenario_folders):
            self.scenario_database.append({})

            # at least 1 cav should show up
            # 用三元运算符来简化判断 (使用 sample 函数代替原先的 shuffle 函数, 因为sample函数有返回值写起来比较统一, 不知道应不影响性能)
            cav_list: List[str] = [cav.name for cav in scenario_folder.iterdir() if cav.is_dir()]
            cav_list = random.sample(cav_list, len(cav_list)) if self.train else sorted(cav_list)
            assert len(cav_list) > 0

            """
            roadside unit data's id is always negative, so here we want to
            make sure they will be in the end of the list as they shouldn't
            be ego vehicle.
            """
            if int(cav_list[0]) < 0:
                cav_list = cav_list[1:] + [cav_list[0]]

            """
            make the first cav to be ego modality
            """
            if self.hetero:
                scenario_name = scenario_folder.stem
                cav_list = self.adaptor.reorder_cav_list(cav_list, scenario_name)

            # loop over all CAV data
            for j, cav_id in enumerate(cav_list):
                if j > self.max_cav - 1:
                    logger.warning(f"In {scenario_folder.stem}, there are too many cavs reinitialize.")
                    break
                self.scenario_database[i][cav_id] = {}

                # save all yaml files to the dictionary
                cav_path = scenario_folder / cav_id

                yaml_files: List[Path] = sorted(file for file in cav_path.glob("*.yaml") if "additional" not in file.stem)
                # this timestamp is not ready
                # fmt: off
                yaml_files = [
                    x for x in yaml_files
                    if not (("2021_08_20_21_10_24" in (path_str := str(x)) and "000265" in path_str) or "2021_09_09_13_20_58" in path_str)
                ]  # fmt: on
                timestamps = [file.stem for file in yaml_files]  # 来自GPT: 把提取 timestamp 函数删掉了 (一行代码完事)

                for timestamp in timestamps:
                    # 将加载数据路径的函数, 移到了一个单独的函数中 (如果因为后面的代码还需要 `lidar_file` 我一定会让 `_GetTimestampDataPath` 函数返回一个字典)
                    yaml_file, lidar_file, camera_files, depth_files = _GetTimestampDataPath(cav_path, timestamp)
                    pfTimestampData = PFTimestampData(
                        yaml=yaml_file, lidar=lidar_file, cameras=camera_files, depths=depth_files
                    )

                    if self.hetero:
                        scenario_name = scenario_folder.stem
                        pfTimestampData.modality_name = self.adaptor.ReassignCAVModality(scenario_name, cav_id, j)
                        pfTimestampData.lidar = self.adaptor.switch_lidar_channels(pfTimestampData.modality_name, lidar_file)

                    # load extra data
                    for file_extension in self.add_data_extension:
                        file_name = os.path.join(cav_path, timestamp + "_" + file_extension)
                        pfTimestampData.file_extension = file_name

                    self.scenario_database[i][cav_id][timestamp] = pfTimestampData
                # Assume all cavs will have the same timestamps length. Thus
                # we only need to calculate for the first vehicle in the
                # scene.
                if j == 0:
                    # we regard the agent with the minimum id as the ego
                    self.scenario_database[i][cav_id]["ego"] = True
                    # 来自GPT: 延迟计算 len_record： 在更新 len_record 时，可以直接将长度累加计算合并到一次操作中，减少冗余代码
                    total_len = self.len_record[-1] if self.len_record else 0
                    self.len_record.append(total_len + len(timestamps))
                else:
                    self.scenario_database[i][cav_id]["ego"] = False
        prefix = "训练集" if self.train else "验证集"
        logger.important(f"{prefix}数据总长度: {self.len_record[-1]}")

    def _GetScenarioIndex(self, idx):
        """Find the correct scenario index based on idx."""
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                return i
        return 0

    def __getitem__(self, idx):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        # we loop the accumulated length list to see get the scenario index
        scenario_index = self._GetScenarioIndex(idx)
        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else idx - self.len_record[scenario_index - 1]

        scenario_database = self.scenario_database[scenario_index]
        # retrieve the corresponding timestamp key
        # 来自GPT, 经过 GPT 优化后的代码, 可能可读性上不是很好 (TODO: 为这一行代码增加一些注释)
        timestamp_key = list(next(iter(scenario_database.values())).items())[timestamp_index][0]
        data: Dict[str, CAVData] = {}
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            cavData = CAVData(
                ego=cav_content["ego"],
                params=_LoadParams(cav_content[timestamp_key].yaml),
                camera_data=load_camera_data(cav_content[timestamp_key].cameras) if self.load_camera_file else [],
                depth_data=load_camera_data(cav_content[timestamp_key].depths) if self.load_depth_file else [],
            )

            # load lidar file
            if self.load_lidar_file or self.visualize:
                cavData.lidar_np = pcd_utils.pcd_to_np(cav_content[timestamp_key].lidar)

            if self.hetero:
                cavData.modality_name = cav_content[timestamp_key].modality_name

            for file_extension in self.add_data_extension:
                # if not find in the current directory
                # go to additional folder
                filePath = cav_content[timestamp_key][file_extension]
                if not os.path.exists(filePath):
                    filePath = _ReplaceWithAdditional(filePath)

                cavData[file_extension] = load_yaml(filePath) if ".yaml" in file_extension else cv2.imread(filePath)
            data[cav_id] = cavData
        return data

    def GetNoisedData(self, idx) -> Dict[str, CAVData]:
        return self.addNoise(self.__getitem__(idx))

    def __len__(self):
        return self.len_record[-1]

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.

        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape

        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw

        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {"lidar_np": lidar_np, "object_bbx_center": object_bbx_center, "object_bbx_mask": object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict["lidar_np"]
        object_bbx_center = tmp_dict["object_bbx_center"]
        object_bbx_mask = tmp_dict["object_bbx_mask"]

        return lidar_np, object_bbx_center, object_bbx_mask

    def GenerateObjectCenterLidar(self, cav_contents: List[CAVData], reference_lidar_pose: np.ndarray):
        return self.post_processor.GenerateObjectCenter(cav_contents, reference_lidar_pose)

    def generate_object_center_lidar(self, cav_contents, reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_object_center(cav_contents, reference_lidar_pose)

    def generate_object_center_camera(self, cav_contents, reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        visibility_map : np.ndarray
            for OPV2V, its 256*256 resolution. 0.39m per pixel. heading up.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_visible_object_center(cav_contents, reference_lidar_pose)

    def get_ext_int(self, params, camera_id):
        """该函数可能会被其他类调用, 所以不可能为静态的"""
        camera_coords = np.array(params["camera%d" % camera_id]["cords"]).astype(np.float32)
        camera_to_lidar = x1_to_x2(camera_coords, params["lidar_pose_clean"]).astype(np.float32)  # T_LiDAR_camera
        camera_to_lidar = camera_to_lidar @ np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32
        )  # UE4 coord to opencv coord
        camera_intrinsic = np.array(params["camera%d" % camera_id]["intrinsic"]).astype(np.float32)
        return camera_to_lidar, camera_intrinsic
