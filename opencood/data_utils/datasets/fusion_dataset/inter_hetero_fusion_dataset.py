"""
-*- coding: utf-8 -*-
Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
License: TDG-Attribution-NonCommercial-NoDistrib

intermediate hetero fusion dataset

Note that for DAIR-V2X dataset,
Each agent should retrieve the objects itself, and merge them by iou,
instead of using the cooperative label.
"""

from typing import Dict

import numpy as np
import pyximport
import torch
from torch.utils.data import Dataset

from opencood.data_utils.data_models.dataset_models import CAVData
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils import box_utils as box_utils
from opencood.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
)
from opencood.utils.common_utils import merge_features_to_dict
from opencood.utils.common_utils import read_json
from opencood.utils.heter_utils import Adaptor
from opencood.utils.pcd_utils import (
    downsample_lidar_minimum,
    ProcessPointsUsingCython,
)


pyximport.install(language_level=3, setup_args={"include_dirs": np.get_include()})
from opencood.utils.cython.transformation_utils import GetPairwiseTransformation, X1ToX2


def _GetEgoCAVInfo(base_data_dict):
    for cav_id, cav_content in base_data_dict.items():
        if cav_content.ego:
            return cav_id, cav_content.params["lidar_pose"], cav_content.params["lidar_pose_clean"], cav_content
    return -1, [], None


class InterHeteroFusionDataset(Dataset):
    """
    删除一些看不懂的标志变量, 以及一些对于其他类型的数据集的处理
    """

    def __init__(self, params, visualize, train, baseDataset):
        super().__init__()
        self.params = params
        self.visualize, self.train, self.hetero = visualize, train, True
        self.comm_range = params["comm_range"]

        # fmt: off
        heteroParams = params["hetero"]
        self.modality_assignment, self.ego_modality, self.modality_name_list = (
            read_json(heteroParams["assignment_path"]), heteroParams["ego_modality"], list(heteroParams["modality_setting"].keys()),
        )
        self.adaptor = Adaptor(
            self.ego_modality, self.modality_name_list, self.modality_assignment,
            heteroParams.get("lidar_channels_dict", {}), heteroParams["mapping_dict"], heteroParams.get("cav_preference", None),
            train,
        )
        # fmt: on
        self.baseDataset = baseDataset(params, visualize, train, self.hetero, self.adaptor)

        self.anchor_box = self.baseDataset.post_processor.GenerateAnchorBox()
        self.anchor_box_torch = torch.from_numpy(self.anchor_box)

        self.sensor_type_dict, self.preprocessor, self.dataAugConf = {}, {}, {}
        for modality_name, modal_setting in params["hetero"]["modality_setting"].items():
            self.sensor_type_dict[modality_name] = modal_setting["sensor_type"]
            match modal_setting["sensor_type"]:
                case "lidar":
                    self.preprocessor[modality_name] = build_preprocessor(modal_setting["preprocess"], train)
                case "camera":
                    self.dataAugConf = modal_setting["data_aug_conf"]
                case _:
                    raise TypeError("Not support this type of sensor")

        self.baseDataset.reinitialize()

    def _ProcessLidarData(self, cavData: CAVData, sensor_type, modality_name, transformation_matrix: np.ndarray[np.float64]):
        lidar_np = ProcessPointsUsingCython(cavData.lidar_np)  # shape: (点云数量, 4)

        selected_cav_processed = {}
        if self.visualize:  # filter lidar
            # 对点云坐标进行投影 (不包括最后一维, 最后一维是反射强度) 为了使用 cython 这里与 `lidar_np` 的 dtype 保持一致
            # project the lidar to ego space x, y, z in ego space
            projected_lidar = box_utils.ProjectPointsByMatrixUsingCython(
                lidar_np[:, :3], transformation_matrix.astype(np.float32)
            )
            selected_cav_processed.update({"projected_lidar": projected_lidar})

        if sensor_type == "lidar":
            processed_lidar = self.preprocessor[modality_name].preprocess(lidar_np)
            selected_cav_processed.update({f"processed_features_{modality_name}": processed_lidar})

        return selected_cav_processed

    def _ProcessCameraData(self, selected_cav_base, modality_name):
        camera_data_list, params = selected_cav_base["camera_data"], selected_cav_base["params"]
        imgs, rots, trans, intrins, extrinsics, post_rots, post_trans = [], [], [], [], [], [], []

        for idx, img in enumerate(camera_data_list):
            camera_to_lidar, camera_intrinsic = self.get_ext_int(params, idx)
            intrin = torch.from_numpy(camera_intrinsic)

            # R_wc, we consider world-coord is the lidar-coord; t_wc
            rot, tran = torch.from_numpy(camera_to_lidar[:3, :3]), torch.from_numpy(camera_to_lidar[:3, 3])
            post_rot, post_tran = torch.eye(2), torch.zeros(2)
            img_src = [img]

            # depth
            if self.load_depth_file:
                depth_img = selected_cav_base["depth_data"][idx]
                img_src.append(depth_img)

            # data augmentation
            resize, resize_dims, crop, flip, rotate = sample_augmentation(
                eval(f"self.data_aug_conf_{modality_name}"), self.train
            )
            img_src, post_rot2, post_tran2 = img_transform(
                img_src, post_rot, post_tran, resize, resize_dims, crop, flip, rotate
            )
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            # decouple RGB and Depth

            img_src[0] = normalize_img(img_src[0])
            if self.load_depth_file:
                img_src[1] = img_to_tensor(img_src[1]) * 255

            imgs.append(img_src)
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return {
            "imgs": torch.stack(imgs),
            "intrins": torch.stack(intrins),
            "extrinsics": torch.stack(extrinsics),
            "rots": torch.stack(rots),
            "trans": torch.stack(trans),
            "post_rots": torch.stack(post_rots),
            "post_trans": torch.stack(post_trans),
        }

    def get_item_single_car(self, cavData: CAVData, ego_pose: np.ndarray[np.float64], ego_pose_clean: np.ndarray[np.float64]):
        """
        Process a single CAV's information for the train/test pipeline.


        Parameters
        ----------
        cavData : dict
            The dictionary contains a single CAV's raw information.
            including 'params', 'camera_data'
        ego_pose : list, length 6
            The ego vehicle lidar pose under world coordinate.
        ego_pose_clean : list, length 6
            only used for gt box generation

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        processedCAVData = {}

        # calculate the transformation matrix 向自车看齐
        transformation_matrix = X1ToX2(cavData.params["lidar_pose"], ego_pose)
        transformation_matrix_clean = X1ToX2(cavData.params["lidar_pose_clean"], ego_pose_clean)

        modality_name = cavData.modality_name
        sensor_type = self.sensor_type_dict[modality_name]

        # lidar
        if sensor_type == "lidar" or self.visualize:
            processedCAVData.update(self._ProcessLidarData(cavData, sensor_type, modality_name, transformation_matrix))

        # generate targets label single GT, note the reference pose is itself.
        object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center([cavData], cavData.params["lidar_pose"])

        label_dict = self.baseDataset.post_processor.GenerateLabel(object_bbx_center, self.anchor_box, object_bbx_mask)
        processedCAVData.update({
            "single_label_dict": label_dict,
            "single_object_bbx_center": object_bbx_center,
            "single_object_bbx_mask": object_bbx_mask,
        })

        # camera
        if sensor_type == "camera":
            processedCAVData.update({f"image_inputs_{modality_name}": self._ProcessCameraData(cavData, modality_name)})

        # anchor box
        processedCAVData.update({"anchor_box": self.anchor_box})

        # note the reference pose ego
        object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center([cavData], ego_pose_clean)

        processedCAVData.update({
            "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
            "object_bbx_mask": object_bbx_mask,
            "object_ids": object_ids,
            "transformation_matrix": transformation_matrix,
            "transformation_matrix_clean": transformation_matrix_clean,
        })

        return processedCAVData

    def _GetLegalCAVIds(self, base_data_dict, ego_lidar_pose):
        """
        获得合法 (在与自车的通信范围内, 有被分配模态) 的车辆 id
        """

        def _GetDistance(lidar_pose: np.ndarray[np.float64]):
            return np.sqrt((lidar_pose[0] - ego_lidar_pose[0]) ** 2 + (lidar_pose[1] - ego_lidar_pose[1]) ** 2)

        def _Judge(modality_name, lidar_pose):
            return not self.adaptor.unmatched_modality(modality_name) and _GetDistance(lidar_pose) <= self.comm_range

        return [cav_id for cav_id, cav in base_data_dict.items() if _Judge(cav.modality_name, cav.params["lidar_pose"])]

    def __getitem__(self, idx):
        base_data_dict: Dict[str, CAVData] = self.baseDataset.GetNoisedData(idx)

        processed_data_dict = {"ego": {}}

        # first find the ego vehicle's lidar pose
        ego_id, ego_pose, ego_pose_clean, ego_cav_base = _GetEgoCAVInfo(base_data_dict)
        legalCAVIdList = self._GetLegalCAVIds(base_data_dict, ego_pose)

        if len(legalCAVIdList) == 0:
            return None

        inputListModalities = {f"m{i}": [] for i in range(4)}  # can contain lidar or camera

        agent_modality_list = []
        object_stack = []
        object_id_stack = []
        single_label_list = []
        single_object_bbx_center_list = []
        single_object_bbx_mask_list = []

        if self.visualize:
            projected_lidar_stack = []
            input_list_m1_proj = []  # 2023.8.31 to correct discretization errors with kd flag
            input_list_m2_proj = []
            input_list_m3_proj = []
            input_list_m4_proj = []

        lidar_pose_clean_list, lidar_pose_list = zip(*[
            (base_data_dict[cav_id].params["lidar_pose_clean"], base_data_dict[cav_id].params["lidar_pose"])
            for cav_id in legalCAVIdList
        ])
        lidar_poses, lidar_poses_clean = np.array(lidar_pose_list).reshape(-1, 6), np.array(lidar_pose_clean_list).reshape(-1, 6) # fmt: skip

        # TODO: 这里或许直接传递 `lidar_poses` 会更好, 但是该怎么做呢?
        pairwise_t_matrix = GetPairwiseTransformation(base_data_dict, self.baseDataset.MaxCAV, False)

        # merge preprocessed features from different cavs into the same dict
        cav_num = len(legalCAVIdList)

        for _i, cav_id in enumerate(legalCAVIdList):
            legalCAVData = base_data_dict[cav_id]
            modality_name = legalCAVData.modality_name
            sensor_type = self.sensor_type_dict[modality_name]

            # TODO: 这个 generate_object_center 的作用是什么呢?
            # dynamic object center generator! for heterogeneous input

            if self.visualize:
                self.generate_object_center = self.baseDataset.GenerateObjectCenter
            elif sensor_type == "lidar":  # TODO: 这里先只是讨论 lidar 的情况
                self.generate_object_center = self.baseDataset.GenerateObjectCenterLidar
            else:
                self.generate_object_center = eval(f"self.generate_object_center_{sensor_type}")

            selected_cav_processed = self.get_item_single_car(legalCAVData, ego_pose, ego_pose_clean)

            object_stack.append(selected_cav_processed["object_bbx_center"])
            object_id_stack += selected_cav_processed["object_ids"]

            match sensor_type:
                case "lidar":
                    inputListModalities[modality_name].append(selected_cav_processed[f"processed_features_{modality_name}"])
                case "camera":
                    inputListModalities[modality_name].append(selected_cav_processed[f"processed_features_{modality_name}"])
                case _:
                    raise TypeError("错误的传感器类型!")

            agent_modality_list.append(modality_name)

            if self.visualize:
                # heterogeneous setting do not support disconet' kd
                projected_lidar_stack.append(selected_cav_processed["projected_lidar"])
                if sensor_type == "lidar" and self.kd_flag:
                    eval(f"input_list_{modality_name}_proj").append(
                        selected_cav_processed[f"processed_features_{modality_name}_proj"]
                    )

            if self.hetero:
                single_label_list.append(selected_cav_processed["single_label_dict"])
                single_object_bbx_center_list.append(selected_cav_processed["single_object_bbx_center"])
                single_object_bbx_mask_list.append(selected_cav_processed["single_object_bbx_mask"])

        # generate single view GT label
        if self.hetero:
            single_label_dicts = self.baseDataset.post_processor.collate_batch(single_label_list)
            single_object_bbx_center = torch.from_numpy(np.array(single_object_bbx_center_list))
            single_object_bbx_mask = torch.from_numpy(np.array(single_object_bbx_mask_list))
            processed_data_dict["ego"].update({
                "single_label_dict_torch": single_label_dicts,
                "single_object_bbx_center_torch": single_object_bbx_center,
                "single_object_bbx_mask_torch": single_object_bbx_mask,
            })

        # exclude all repetitive objects, OPV2V-H
        unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = np.zeros((self.params["postprocess"]["max_num"], 7))
        mask = np.zeros(self.params["postprocess"]["max_num"], dtype=np.int64)
        object_bbx_center[: object_stack.shape[0], :] = object_stack
        mask[: object_stack.shape[0]] = 1

        for modality_name in self.modality_name_list:
            if self.sensor_type_dict[modality_name] == "lidar":
                merged_feature_dict = merge_features_to_dict(inputListModalities[modality_name])
                processed_data_dict["ego"].update({f"input_{modality_name}": merged_feature_dict})  # maybe None
            elif self.sensor_type_dict[modality_name] == "camera":
                merged_image_inputs_dict = merge_features_to_dict(eval(f"input_list_{modality_name}"), merge="stack")
                processed_data_dict["ego"].update({f"input_{modality_name}": merged_image_inputs_dict})  # maybe None

        processed_data_dict["ego"].update({"agent_modality_list": agent_modality_list})

        # generate targets label
        label_dict = self.baseDataset.post_processor.GenerateLabel(object_bbx_center, self.anchor_box, mask)

        processed_data_dict["ego"].update({
            "object_bbx_center": object_bbx_center,
            "object_bbx_mask": mask,
            "object_ids": [object_id_stack[i] for i in unique_indices],
            "anchor_box": self.anchor_box,
            "label_dict": label_dict,
            "cav_num": cav_num,
            "pairwise_t_matrix": pairwise_t_matrix,
            "lidar_poses_clean": lidar_poses_clean,
            "lidar_poses": lidar_poses,
        })

        # TODO: 我不太明白这里为什么会将 `sample_idx` 和 `cav_id_list` 单独写到后面添加, 是为了保持 processed_data_dict 字典顺序?
        if self.visualize:
            processed_data_dict["ego"].update({"origin_lidar": np.vstack(projected_lidar_stack)})

        processed_data_dict["ego"].update({"sample_idx": idx, "cav_id_list": legalCAVIdList})

        return processed_data_dict

    def collate_batch_train(self, batch):
        # Intermediate fusion is different the other two
        output_dict = {"ego": {}}

        object_bbx_center = []
        object_bbx_mask = []
        object_ids = []
        inputsListModalities = {f"m{i}": [] for i in range(4)}

        agent_modality_list = []
        # used to record different scenario
        record_len = []
        label_dict_list = []
        lidar_pose_list = []
        origin_lidar = []
        lidar_pose_clean_list = []

        # pairwise transformation matrix
        pairwise_t_matrix_list = []

        ### 2022.10.10 single gt ####
        if self.hetero:
            pos_equal_one_single = []
            neg_equal_one_single = []
            targets_single = []
            object_bbx_center_single = []
            object_bbx_mask_single = []

        for i in range(len(batch)):
            ego_dict = batch[i]["ego"]
            object_bbx_center.append(ego_dict["object_bbx_center"])
            object_bbx_mask.append(ego_dict["object_bbx_mask"])
            object_ids.append(ego_dict["object_ids"])
            lidar_pose_list.append(ego_dict["lidar_poses"])  # ego_dict['lidar_pose'] is np.ndarray [N,6]
            lidar_pose_clean_list.append(ego_dict["lidar_poses_clean"])

            for modality_name in self.modality_name_list:
                if ego_dict[f"input_{modality_name}"] is not None:
                    inputsListModalities[modality_name].append(ego_dict[f"input_{modality_name}"])  # {} if empty?

            agent_modality_list.extend(ego_dict["agent_modality_list"])

            record_len.append(ego_dict["cav_num"])
            label_dict_list.append(ego_dict["label_dict"])
            pairwise_t_matrix_list.append(ego_dict["pairwise_t_matrix"])

            if self.visualize:
                origin_lidar.append(ego_dict["origin_lidar"])

            ### 2022.10.10 single gt ####
            if self.hetero:
                pos_equal_one_single.append(ego_dict["single_label_dict_torch"]["pos_equal_one"])
                neg_equal_one_single.append(ego_dict["single_label_dict_torch"]["neg_equal_one"])
                targets_single.append(ego_dict["single_label_dict_torch"]["targets"])
                object_bbx_center_single.append(ego_dict["single_object_bbx_center_torch"])
                object_bbx_mask_single.append(ego_dict["single_object_bbx_mask_torch"])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        # 2023.2.5
        for modality_name in self.modality_name_list:
            if len(inputsListModalities[modality_name]) != 0:
                if self.sensor_type_dict[modality_name] == "lidar":
                    merged_feature_dict = merge_features_to_dict(inputsListModalities[modality_name])
                    processed_lidar_torch_dict = self.preprocessor[modality_name].collate_batch(merged_feature_dict)
                    output_dict["ego"].update({f"inputs_{modality_name}": processed_lidar_torch_dict})

                elif self.sensor_type_dict[modality_name] == "camera":
                    merged_image_inputs_dict = merge_features_to_dict(eval(f"inputs_list_{modality_name}"), merge="cat")
                    output_dict["ego"].update({f"inputs_{modality_name}": merged_image_inputs_dict})

        output_dict["ego"].update({"agent_modality_list": agent_modality_list})

        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
        lidar_pose_clean = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))
        label_torch_dict = self.baseDataset.post_processor.collate_batch(label_dict_list)

        # for centerpoint
        label_torch_dict.update({"object_bbx_center": object_bbx_center, "object_bbx_mask": object_bbx_mask})

        # (B, max_cav)
        pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

        # add pairwise_t_matrix to label dict
        label_torch_dict["pairwise_t_matrix"] = pairwise_t_matrix
        label_torch_dict["record_len"] = record_len

        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict["ego"].update({
            "object_bbx_center": object_bbx_center,
            "object_bbx_mask": object_bbx_mask,
            "record_len": record_len,
            "label_dict": label_torch_dict,
            "object_ids": object_ids[0],
            "pairwise_t_matrix": pairwise_t_matrix,
            "lidar_pose_clean": lidar_pose_clean,
            "lidar_pose": lidar_pose,
            "anchor_box": self.anchor_box_torch,
        })

        if self.visualize:
            origin_lidar = np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict["ego"].update({"origin_lidar": origin_lidar})

        if self.hetero:
            output_dict["ego"].update({
                "label_dict_single": {
                    "pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
                    "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
                    "targets": torch.cat(targets_single, dim=0),
                    # for centerpoint
                    "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                    "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0),
                },
                "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0),
            })

        return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        if batch[0] is None:
            return None
        output_dict = self.collate_batch_train(batch)
        if output_dict is None:
            return None

        # check if anchor box in the batch
        if batch[0]["ego"]["anchor_box"] is not None:
            output_dict["ego"].update({"anchor_box": self.anchor_box_torch})

        # save the transformation matrix (4, 4) to ego vehicle
        # transformation is only used in post process (no use.)
        # we all predict boxes in ego coord.
        transformation_matrix_torch = torch.from_numpy(np.identity(4)).float()
        transformation_matrix_clean_torch = torch.from_numpy(np.identity(4)).float()

        output_dict["ego"].update({
            "transformation_matrix": transformation_matrix_torch,
            "transformation_matrix_clean": transformation_matrix_clean_torch,
        })

        output_dict["ego"].update({
            "sample_idx": batch[0]["ego"]["sample_idx"],
            "cav_id_list": batch[0]["ego"]["cav_id_list"],
            "agent_modality_list": batch[0]["ego"]["agent_modality_list"],
        })

        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor
