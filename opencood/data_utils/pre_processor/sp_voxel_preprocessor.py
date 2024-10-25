# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Transform points to voxels using sparse conv library
"""
import sys

import numpy as np
import torch
from cumm import tensorview as tv
from spconv.utils import Point2VoxelCPU3d as VoxelGenerator

from opencood.data_utils.pre_processor.base_preprocessor import BasePreprocessor


class SpVoxelPreprocessor(BasePreprocessor):
    def __init__(self, preprocess_params, train):
        super().__init__(preprocess_params, train)

        self.lidar_range = self.params["cav_lidar_range"]
        self.voxel_size = self.params["args"]["voxel_size"]
        self.max_points_per_voxel = self.params["args"]["max_points_per_voxel"]

        self.max_voxels = self.params["args"]["max_voxel_train"] if train else self.params["args"]["max_voxel_test"]

        grid_size = (np.array(self.lidar_range[3:6]) - np.array(self.lidar_range[0:3])) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)

        # use sparse conv library to generate voxel TODO: 变量的命名上是否要做到和使用的 api 保持一致呢?
        self.voxel_generator = VoxelGenerator(self.voxel_size, self.lidar_range, 4, self.max_voxels, self.max_points_per_voxel)

    def preprocess(self, pcd_np: np.ndarray[np.float32]):
        # 必须使用 `tv.from_numpy`, 因为 `point_to_voxel` 会对输入进行类型检查, tv 版本返回的和 torch 返回的不一样
        pcd_tv = tv.from_numpy(pcd_np)
        voxel_output = self.voxel_generator.point_to_voxel(pcd_tv)
        voxels, coordinates, num_points = voxel_output

        return {"voxel_features": voxels.numpy(), "voxel_coords": coordinates.numpy(), "voxel_num_points": num_points.numpy()}

    def collate_batch(self, batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list or dict
            List or dictionary.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """

        if isinstance(batch, list):
            return self.collate_batch_list(batch)
        elif isinstance(batch, dict):
            return self.collate_batch_dict(batch)
        else:
            sys.exit("Batch has too be a list or a dictionarn")

    @staticmethod
    def collate_batch_list(batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list
            List of dictionary. Each dictionary represent a single frame.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        voxel_features, voxel_num_points, voxel_coords = [], [], []

        for batchData in batch:
            voxel_features.append(batchData["voxel_features"])
            voxel_num_points.append(batchData["voxel_num_points"])
            coords = batchData["voxel_coords"]
            voxel_coords.append(np.pad(coords, ((0, 0), (1, 0)), mode="constant", constant_values=i))

        voxel_num_points = torch.from_numpy(np.concatenate(voxel_num_points))
        voxel_features = torch.from_numpy(np.concatenate(voxel_features))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        return {"voxel_features": voxel_features, "voxel_coords": voxel_coords, "voxel_num_points": voxel_num_points}

    @staticmethod
    def collate_batch_dict(batch: dict):
        """
        Collate batch if the batch is a dictionary,
        eg: {'voxel_features': [feature1, feature2...., feature n]}

        Parameters
        ----------
        batch : dict

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        voxel_features = torch.from_numpy(np.concatenate(batch["voxel_features"]))
        voxel_num_points = torch.from_numpy(np.concatenate(batch["voxel_num_points"]))
        voxel_coords = torch.from_numpy(
            np.concatenate([
                np.pad(coord, ((0, 0), (1, 0)), mode="constant", constant_values=i)
                for i, coord in enumerate(batch["voxel_coords"])
            ])
        )

        return {"voxel_features": voxel_features, "voxel_coords": voxel_coords, "voxel_num_points": voxel_num_points}


def _TestForFromNumpy():
    """
    经测试 `tv.from_nupy` 速度更快. 但是 cumm 这个库网上的资料挺少的.
    """
    import time

    pcd_np = np.random.rand(3000, 4)

    # Test for `tv.from_numpy`
    start_tv = time.time()
    a = tv.from_numpy(pcd_np)
    end_tv = time.time()
    print("tv.from_numpy:    ", end_tv - start_tv)

    start_ch = time.time()
    b = torch.from_numpy(pcd_np)
    end_ch = time.time()
    print("torch.from_numpy: ", end_ch - start_ch)


if __name__ == "__main__":
    _TestForFromNumpy()
