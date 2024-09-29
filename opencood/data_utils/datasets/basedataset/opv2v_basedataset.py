# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import json
import os
import random
from collections import OrderedDict
from pathlib import Path

import cv2
import h5py
import numpy as np
import opencood.utils.pcd_utils as pcd_utils
from PIL import Image
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.camera_utils import load_camera_data
from opencood.utils.transformation_utils import x1_to_x2
from torch.utils.data import Dataset


def _GetTimestampDataPath(cavPath, timestamp):
    """
    获取某一时间戳下数据的路径

    :param cavPath 汽车所在路径
    :param timestamp 时间戳
    """
    yaml_file = os.path.join(cavPath, timestamp + ".yaml")
    lidar_file = os.path.join(cavPath, timestamp + ".pcd")
    camera_files = [Path(cavPath) / f"{timestamp}_camera{i}.png" for i in range(4)]
    depth_files = [Path(cavPath) / f"{timestamp}_depth{i}.png" for i in range(4)]
    depth_files = [depth_file.replace("OPV2V", "OPV2V_Hetero") for depth_file in depth_files]
    return yaml_file, lidar_file, camera_files, depth_files


class OPV2VBaseDataset(Dataset):
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train
        self.use_hdf5 = True

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)
        if "data_augment" in params:  # late and early
            self.data_augmentor = DataAugmentor(params["data_augment"], train)
        else:  # intermediate
            self.data_augmentor = None

        if self.train:
            root_dir = params["root_dir"]
        else:
            root_dir = params["validate_dir"]
        self.root_dir = root_dir

        print("Dataset dir:", root_dir)

        # 来自GPT: 冗余的默认值处理
        self.max_cav = params.get("train_params", {}).get("max_cav", 5)

        # 来自 GPT, 条件判断的简化
        self.load_lidar_file = "lidar" in params["input_source"] or self.visualize
        self.load_camera_file = "camera" in params["input_source"]
        self.load_depth_file = "depth" in params["input_source"]

        self.label_type = params["label_type"]  # 'lidar' or 'camera'
        self.generate_object_center = (
            self.generate_object_center_lidar if self.label_type == "lidar" else self.generate_object_center_camera
        )
        self.generate_object_center_single = (
            self.generate_object_center
        )  # will it follows 'self.generate_object_center' when 'self.generate_object_center' change?

        if self.load_camera_file:
            self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        # by default, we load lidar, camera and metadata. But users may
        # define additional inputs/tasks
        self.add_data_extension = params["add_data_extension"] if "add_data_extension" in params else []

        if "noise_setting" not in self.params:
            # 来自GPT: 如果不需要特定顺序, 建议使用普通字典 `{}`, 因为自 python3.7 开始, 普通字典就已经是有序的了.
            # WARNING: 本文件中所有使用 OrderedDict 的地方, 均被替换为 `{}`
            self.params["noise_setting"] = {}
            self.params["noise_setting"]["add_noise"] = False

        # first load all paths of different scenarios
        # 来自GPT: 路径处理的优化
        self.scenario_folders = sorted([folder for folder in Path(root_dir).iterdir() if folder.is_dir()])

        self.reinitialize()

    def reinitialize(self):
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database = {}
        self.len_record = []

        # loop over all scenarios
        for i, scenario_folder in enumerate(self.scenario_folders):
            self.scenario_database.update({i: {}})

            # at least 1 cav should show up
            # 用三元运算符来简化判断 (使用 sample 函数代替原先的 shuffle 函数, 因为sample函数有返回值写起来比较统一, 不知道应不影响性能)
            cav_list = [cav.name for cav in Path(scenario_folder).iterdir() if cav.is_dir()]
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
            if getattr(self, "heterogeneous", False):
                scenario_name = scenario_folder.split("/")[-1]
                cav_list = self.adaptor.reorder_cav_list(cav_list, scenario_name)

            # loop over all CAV data
            for j, cav_id in enumerate(cav_list):
                if j > self.max_cav - 1:
                    print("too many cavs reinitialize")
                    break
                self.scenario_database[i][cav_id] = {}

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                yaml_files = sorted([str(file) for file in Path(cav_path).glob("*.yaml") if "additional" not in file.stem])

                # this timestamp is not ready
                yaml_files = [x for x in yaml_files if not ("2021_08_20_21_10_24" in x and "000265" in x)]

                timestamps = [Path(file).stem for file in yaml_files]  # 来自GPT: 把提取 timestamp 函数删掉了 (一行代码完事)

                for timestamp in timestamps:
                    # 将加载数据路径的函数, 移到了一个单独的函数中 (如果因为后面的代码还需要 `lidar_file` 我一定会让 `_GetTimestampDataPath` 函数返回一个字典)
                    yaml_file, lidar_file, camera_files, depth_files = _GetTimestampDataPath(cav_path, timestamp)
                    self.scenario_database[i][cav_id][timestamp] = {
                        "yaml": yaml_file,
                        "lidar": lidar_file,
                        "cameras": camera_files,
                        "depths": depth_files,
                    }

                    if getattr(self, "heterogeneous", False):
                        scenario_name = scenario_folder.split("/")[-1]

                        cav_modality = self.adaptor.reassign_cav_modality(self.modality_assignment[scenario_name][cav_id], j)

                        self.scenario_database[i][cav_id][timestamp].update(
                            {"modality": cav_modality, "lidar": self.adaptor.switch_lidar_channels(cav_modality, lidar_file)}
                        )

                    # load extra data
                    for file_extension in self.add_data_extension:
                        file_name = os.path.join(cav_path, timestamp + "_" + file_extension)

                        self.scenario_database[i][cav_id][timestamp][file_extension] = file_name

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
        print("len:", self.len_record[-1])

    def retrieve_base_data(self, idx):
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
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else idx - self.len_record[scenario_index - 1]
        # retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database, timestamp_index)
        data = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            data[cav_id] = OrderedDict()
            data[cav_id]["ego"] = cav_content["ego"]

            # load param file: json is faster than yaml
            json_file = cav_content[timestamp_key]["yaml"].replace("yaml", "json")
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    data[cav_id]["params"] = json.load(f)
            else:
                data[cav_id]["params"] = load_yaml(cav_content[timestamp_key]["yaml"])

            # load camera file: hdf5 is faster than png
            hdf5_file = cav_content[timestamp_key]["cameras"][0].replace("camera0.png", "imgs.hdf5")

            if self.use_hdf5 and os.path.exists(hdf5_file):
                with h5py.File(hdf5_file, "r") as f:
                    data[cav_id]["camera_data"] = []
                    data[cav_id]["depth_data"] = []
                    for i in range(4):
                        if self.load_camera_file:
                            data[cav_id]["camera_data"].append(Image.fromarray(f[f"camera{i}"][()]))
                        if self.load_depth_file:
                            data[cav_id]["depth_data"].append(Image.fromarray(f[f"depth{i}"][()]))
            else:
                if self.load_camera_file:
                    data[cav_id]["camera_data"] = load_camera_data(cav_content[timestamp_key]["cameras"])
                if self.load_depth_file:
                    data[cav_id]["depth_data"] = load_camera_data(cav_content[timestamp_key]["depths"])

            # load lidar file
            if self.load_lidar_file or self.visualize:
                data[cav_id]["lidar_np"] = pcd_utils.pcd_to_np(cav_content[timestamp_key]["lidar"])

            if getattr(self, "heterogeneous", False):
                data[cav_id]["modality_name"] = cav_content[timestamp_key]["modality_name"]

            for file_extension in self.add_data_extension:
                # if not find in the current directory
                # go to additional folder
                if not os.path.exists(cav_content[timestamp_key][file_extension]):
                    cav_content[timestamp_key][file_extension] = cav_content[timestamp_key][file_extension].replace(
                        "train", "additional/train"
                    )
                    cav_content[timestamp_key][file_extension] = cav_content[timestamp_key][file_extension].replace(
                        "validate", "additional/validate"
                    )
                    cav_content[timestamp_key][file_extension] = cav_content[timestamp_key][file_extension].replace(
                        "test", "additional/test"
                    )

                if ".yaml" in file_extension:
                    data[cav_id][file_extension] = load_yaml(cav_content[timestamp_key][file_extension])
                else:
                    data[cav_id][file_extension] = cv2.imread(cav_content[timestamp_key][file_extension])

        return data

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        pass

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # get all timestamp keys
        timestamp_keys = list(scenario_database.items())[0][1]
        # retrieve the correct index
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

        return timestamp_key

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
        camera_coords = np.array(params["camera%d" % camera_id]["cords"]).astype(np.float32)
        camera_to_lidar = x1_to_x2(camera_coords, params["lidar_pose_clean"]).astype(np.float32)  # T_LiDAR_camera
        camera_to_lidar = camera_to_lidar @ np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32
        )  # UE4 coord to opencv coord
        camera_intrinsic = np.array(params["camera%d" % camera_id]["intrinsic"]).astype(np.float32)
        return camera_to_lidar, camera_intrinsic
