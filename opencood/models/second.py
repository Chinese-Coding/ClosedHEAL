# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch.nn as nn

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x


class Second(nn.Module):
    def __init__(self, args):
        super(Second, self).__init__()

        # mean_vfe
        self.mean_vfe = MeanVFE(args["mean_vfe"], 4)
        # sparse 3d backbone
        self.backbone_3d = VoxelBackBone8x(args["backbone_3d"], 4, args["grid_size"])
        # height compression
        self.height_compression = HeightCompression(args["height_compression"])
        # base ben backbone
        self.backbone_2d = BaseBEVBackbone(args["base_bev_backbone"], 256)

        # head
        self.cls_head = nn.Conv2d(256 * 2, args["anchor_number"], kernel_size=1)
        self.reg_head = nn.Conv2d(256 * 2, 7 * args["anchor_num"], kernel_size=1)

    def forward(self, data_dict):

        voxel_features = data_dict["processed_lidar"]["voxel_features"]
        voxel_coords = data_dict["processed_lidar"]["voxel_coords"]
        voxel_num_points = data_dict["processed_lidar"]["voxel_num_points"]
        batch_size = voxel_coords[:, 0].max() + 1  # batch size is padded in the first idx

        batch_dict = {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_num_points": voxel_num_points,
            "batch_size": batch_size,
        }

        batch_dict = self.mean_vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.height_compression(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)

        spatial_features_2d = batch_dict["spatial_features_2d"]

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        output_dict = {"psm": psm, "rm": rm}

        return output_dict
