"""Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception
"""

import torch
import torch.nn as nn
import torchvision

from opencood.models.fusion_models.pyramid_fusion import PyramidFusion
from opencood.models.hetero_encoders import encoders
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.submodels.resnet_bev_backbone import ResnetBEVBackbone
from opencood.utils.model_utils import check_trainable_module
from opencood.utils.transformation_utils import normalize_pairwise_tfm


class HeteroPyramidCollab(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.modality_name_list = [x for x in params.keys() if x.startswith("m") and x[1:].isdigit()]

        self.cav_range = params["lidar_range"]
        self.sensor_type_dict = {}
        self.cam_crop_info, self.cropRatioW, self.cropRatioH, self.xDist, self.yDist = {}, {}, {}, {}, {}

        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = params[modality_name]
            sensor_name = model_setting["sensor_type"]
            self.sensor_type_dict[modality_name] = sensor_name

            """以下的三个东西都需要体现模态的保存, 所以不使用 dict 进行存储"""
            """Encoder building"""
            encoder_class = encoders[model_setting["core_method"]]
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting["encoder_args"]))
            setattr(self, f"depth_supervision_{modality_name}", model_setting["encoder_args"].get("depth_supervision", False))

            """Backbone building"""
            setattr(self, f"backbone_{modality_name}", ResnetBEVBackbone(model_setting["backbone_args"]))

            """Aligner building"""
            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting["aligner_args"]))

            # 和图像有关的东西
            if sensor_name == "camera":
                grid_conf = model_setting["camera_mask_args"]["grid_conf"]

                self.cropRatioW[modality_name] = self.cav_range[3] / grid_conf["xbound"][1]
                self.cropRatioH[modality_name] = self.cav_range[4] / grid_conf["ybound"][1]

                self.xDist[modality_name] = grid_conf["xbound"][1] - grid_conf["xbound"][0]
                self.yDist[modality_name] = grid_conf["ybound"][1] - grid_conf["ybound"][0]

                self.cam_crop_info[modality_name] = {
                    f"crop_ratio_W_{modality_name}": self.cropRatioW[modality_name],
                    f"crop_ratio_H_{modality_name}": self.cropRatioH[modality_name],
                }

        """For feature transformation"""
        self.H = self.cav_range[4] - self.cav_range[1]
        self.W = self.cav_range[3] - self.cav_range[0]
        self.fake_voxel_size = 1

        """
        Fusion, by default multiscale fusion: 
        Note the input of PyramidFusion has downsampled 2x. (SECOND required)
        """
        self.pyramid_backbone = PyramidFusion(params["fusion_backbone"])

        """Shrink header"""
        self.shrink_flag = False
        if "shrink_header" in params:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(params["shrink_header"])

        """Shared Heads"""
        self.cls_head = nn.Conv2d(params["in_head"], params["anchor_number"], kernel_size=1)
        self.reg_head = nn.Conv2d(params["in_head"], 7 * params["anchor_number"], kernel_size=1)
        self.dir_head = nn.Conv2d(
            params["in_head"], params["dir_args"]["num_bins"] * params["anchor_number"], kernel_size=1
        )  # BIN_NUM = 2

        # compressor will be only trainable
        self.compress = "compressor" in params
        if self.compress:
            self.compressor = NaiveCompressor(params["compressor"]["input_dim"], params["compressor"]["compress_ratio"])

        self.model_train_init()
        # check again which module is not fixed.
        check_trainable_module(self)

    def model_train_init(self):
        # if compress, only make compressor trainable
        if self.compress:
            # freeze all
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            # unfreeze compressor
            self.compressor.train()
            for p in self.compressor.parameters():
                p.requires_grad_(True)

    def forward(self, data_dict):
        output_dict = {"pyramid": "collab", "processed_features": {}}
        agent_modality_list = data_dict["agent_modality_list"]
        affine_matrix = normalize_pairwise_tfm(data_dict["pairwise_t_matrix"], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict["record_len"]
        agentModalitySet = set(agent_modality_list)
        modality_feature_dict = {}
        for modality_name in self.modality_name_list:
            if modality_name not in agentModalitySet:
                continue  # 条件反转, 减少缩进
            # 多视角下多模态如何比较剩下的信息的多少呢? 怎么比较, 没法比较吧
            feature = eval(f"self.encoder_{modality_name}")(data_dict[f"inputs_{modality_name}"])
            feature = eval(f"self.backbone_{modality_name}")(feature)
            feature = eval(f"self.aligner_{modality_name}")(feature)

            """Crop/Padd camera feature map."""
            if self.sensor_type_dict[modality_name] == "camera":
                _, _, H, W = feature.shape
                # 裁剪到和点云采集到的特征图 H, W 一致
                target_H, target_W = int(H * self.cropRatioH[modality_name]), int(W * self.cropRatioW[modality_name])

                crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                feature = crop_func(feature)
                if eval(f"self.depth_supervision_{modality_name}"):
                    output_dict.update({f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items})
            modality_feature_dict[modality_name] = feature

            # 对用于计算共同特征和私有特征的 Loss 的 feature 进行一个简单地裁切
            crop_func = torchvision.transforms.CenterCrop((128, 128))
            output_dict[f"processed_features"][modality_name] = crop_func(feature)

        """Assemble hetero features"""
        counting_dict = {modality_name: 0 for modality_name in self.modality_name_list}
        hetero_feature_2d_list = []
        for modality_name in agent_modality_list:
            hetero_feature_2d_list.append(modality_feature_dict[modality_name][counting_dict[modality_name]])
            counting_dict[modality_name] += 1
        hetero_feature_2d = torch.stack(hetero_feature_2d_list)
        hetero_feature_2d = self.compressor(hetero_feature_2d) if self.compress else hetero_feature_2d

        # hetero_feature_2d is downsampled 2x
        # add croping information to collaboration module

        fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
            hetero_feature_2d, record_len, affine_matrix, agent_modality_list, self.cam_crop_info
        )

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        output_dict.update({
            "cls_preds": self.cls_head(fused_feature),
            "reg_preds": self.reg_head(fused_feature),
            "dir_preds": self.dir_head(fused_feature),
            "occ_single_list": occ_outputs,
        })

        return output_dict
