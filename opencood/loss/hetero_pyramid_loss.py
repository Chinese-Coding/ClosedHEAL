# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
import random
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from opencood.loss.point_pillar_depth_loss import PointPillarDepthLoss
from opencood.loss.point_pillar_loss import sigmoid_focal_loss


def _DecoupleMatrixDiag(matrix):
    nonDiagMask = ~torch.eye(matrix.size(0), dtype=bool)
    try:
        return torch.diag(matrix), matrix[nonDiagMask]
    except IndexError:
        breakpoint()
        pass


def _CalcCLoss(CMatrix, lambdaC):
    diagElements, nonDiagElements = _DecoupleMatrixDiag(CMatrix)
    return (diagElements - 1).pow(2).sum() + lambdaC * nonDiagElements.pow(2).sum()


def _CalaULoss(matrix, lambdaU):
    diagElements, nonDiagElements = _DecoupleMatrixDiag(matrix)
    return diagElements.pow(2).sum() + lambdaU * nonDiagElements.pow(2).sum()


class HeteroPyramidLoss(PointPillarDepthLoss):
    def __init__(self, args: Dict):
        super().__init__(args)
        self.pyramid = args["pyramid"]

        # relative downsampled GT cls map from fused labels.
        self.relative_downsample = self.pyramid["relative_downsample"]
        self.pyramid_weight = self.pyramid["weight"]
        self.num_levels = len(self.relative_downsample)
        self.cuRatio = {}

    """
    # H, W = v1.shape[2:]
                    # v1, v2 = v1.reshape(-1, H, W), v2.reshape(-1, H, W)
                    # for subv1, subv2 in zip(v1, v2):
                    #     subv1, subv2 = subv1.reshape(-1), subv2.reshape(-1)
                    #     featureMatrix = torch.einsum("i, j -> ij", subv1, subv2) / batch_size
                    #     CFeatureLen = int(v1.shape[1] * self.cuRatio[k1 + k2])
                    #     CMatrix, UMatrix = (
                    #         featureMatrix[:CFeatureLen, :CFeatureLen],
                    #         featureMatrix[CFeatureLen:, CFeatureLen:],
                    #     )
                    #     loss += _CalcCLoss(CMatrix, 1) + _CalaULoss(UMatrix, 1)
                    直接展成一维向量计算需要的内存过多, 就算使用 CPU 计算也不可能
                    # v1, v2 = v1.reshape(-1), v2.reshape(-1)
                    # featureMatrix = v1.outer(v2) / batch_size  # Tried to allocate 65536.00 GiB.

                    # CFeatureLen = int(v1.shape[1] * self.cuRatio[k1 + k2])
                    # CMatrix, UMatrix = (featureMatrix[:CFeatureLen, :CFeatureLen], featureMatrix[CFeatureLen:, CFeatureLen:])
                    # loss += _CalcCLoss(CMatrix, 1) + _CalaULoss(UMatrix, 1)
    """

    def _CalcCULoss(self, processed_features: Dict[str, Tensor], batch_size):
        """CU: common adn unique"""
        loss = 0
        # 这样循环会有重复计算的情况出现, 但考虑到循环量不是很大, 这里就先这么写了
        for k1, v1 in processed_features.items():
            for k2, v2 in processed_features.items():
                if k1 != k2 and v1.size(0) == v2.size(0):
                    if self.cuRatio.get(k1 + k2, None) is None or self.cuRatio.get(k2 + k1, None) is None:
                        self.cuRatio[k1 + k2] = self.cuRatio[k2 + k1] = random.random()
                    gap1, gap2 = torch.nn.AdaptiveAvgPool2d(1)(v1), torch.nn.AdaptiveAvgPool2d(1)(v2)
                    # 同一场景下不同模态车的数量可能还不尽相同, shape[0] 的值不可能一直为 1
                    # 还有可能在某一场景下只有一种模态, 加一个判断语句 `and v1.size(0) == v2.size(0)`
                    # 经过一个全局平均池化层就变成了一个 [1, 64, 1, 1] 的张量, 直接展平后算外积
                    gap1, gap2 = gap1.reshape(-1), gap2.reshape(-1)

                    featureMatrix = gap1.outer(gap2) / batch_size  # Tried to allocate 65536.00 GiB.
                    CFeatureLen = int(gap1.shape[0] * self.cuRatio[k1 + k2])
                    CMatrix, UMatrix = (featureMatrix[:CFeatureLen, :CFeatureLen], featureMatrix[CFeatureLen:, CFeatureLen:])
                    loss += _CalcCLoss(CMatrix, 1) + _CalaULoss(UMatrix, 1)
        return loss

    def forward(self, output_dict, target_dict, suffix=""):
        if output_dict["pyramid"] == "collab":  # intermediate fusion, pyramid collab.
            return self.forward_collab(output_dict, target_dict, suffix)

        elif output_dict["pyramid"] == "single":  # late fusion, pyramid single
            return self.forward_single(output_dict, target_dict, suffix)
        raise

    def forward_single(self, output_dict, target_dict, suffix):
        """for hetero_pyramid_single"""
        batch_size = target_dict["pos_equal_one"].shape[0]
        total_loss = super().forward(output_dict, target_dict, suffix)

        occ_single_list = output_dict["occ_single_list"]
        occ_loss = self.calc_occ_loss(occ_single_list, target_dict["pos_equal_one"], target_dict["neg_equal_one"], batch_size)
        total_loss += occ_loss
        self.loss_dict.update({"pyramid_loss": occ_loss.item(), "total_loss": total_loss.item()})
        return total_loss

    def forward_collab(self, output_dict, target_dict, suffix):
        """for hetero_pyramid_collab"""
        if suffix == "":
            total_loss = super().forward(output_dict, target_dict)
            processed_features = output_dict["processed_features"]
            cu_loss = self._CalcCULoss(processed_features, target_dict["pos_equal_one"].shape[0])
            self.loss_dict.update({"cu_loss": 0 if cu_loss == 0 else cu_loss.item()})
            total_loss += cu_loss
            return total_loss

        assert suffix == "_single"
        batch_size = target_dict["pos_equal_one"].shape[0]
        positives, negatives = target_dict["pos_equal_one"], target_dict["neg_equal_one"]
        occ_single_list = output_dict["occ_single_list"]
        occ_loss = self.calc_occ_loss(occ_single_list, positives, negatives, batch_size)
        total_loss = occ_loss
        self.loss_dict = {"pyramid_loss": occ_loss.item(), "total_loss": total_loss.item()}

        return total_loss

    def calc_occ_loss(self, occ_single_list, positives, negatives, batch_size):
        total_occ_loss = 0
        occ_positives = torch.logical_or(positives[..., 0], positives[..., 1]).unsqueeze(-1).float()  # N, H, W
        occ_negatives = torch.logical_and(negatives[..., 0], negatives[..., 1]).unsqueeze(-1).float()  # N, H, W

        for i, occ_preds_single in enumerate(occ_single_list):
            """
            occ_preds_single: N, 1, H, W

            occ_positives: N, H, W, 1
            occ_negatives: N, H, W, 1

            """

            positives_level = F.max_pool2d(occ_positives.permute(0, 3, 1, 2), kernel_size=self.relative_downsample[i]).permute(
                0, 2, 3, 1
            )
            negatives_level = 1 - F.max_pool2d(
                (1 - occ_negatives).permute(0, 3, 1, 2), kernel_size=self.relative_downsample[i]
            ).permute(0, 2, 3, 1)

            occ_labls = positives_level.view(batch_size, -1, 1)
            positives_level = occ_labls
            negatives_level = negatives_level.view(batch_size, -1, 1)

            pos_normalizer = positives_level.sum(1, keepdim=True).float()

            occ_preds = occ_preds_single.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)
            occ_weights = positives_level * self.pos_cls_weight + negatives_level * 1.0
            occ_weights /= torch.clamp(pos_normalizer, min=1.0)
            occ_loss = sigmoid_focal_loss(occ_preds, occ_labls, weights=occ_weights, **self.cls)
            occ_loss = occ_loss.sum() / batch_size
            occ_loss *= self.pyramid_weight[i]

            total_occ_loss += occ_loss

        return total_occ_loss

    def logging(self, epoch, batch_id, batch_len, writer=None, suffix=""):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict.get("total_loss", 0)
        reg_loss = self.loss_dict.get("reg_loss", 0)
        cls_loss = self.loss_dict.get("cls_loss", 0)
        dir_loss = self.loss_dict.get("dir_loss", 0)
        iou_loss = self.loss_dict.get("iou_loss", 0)
        depth_loss = self.loss_dict.get("depth_loss", 0)
        pyramid_loss = self.loss_dict.get("pyramid_loss", 0)
        cu_loss = self.loss_dict.get("cu_loss", 0)

        print(
            f"[epoch {epoch}][{batch_id + 1}/{batch_len}]{suffix} || "
            f"Loss: {total_loss:.4f} || Conf Loss: {cls_loss:.4f} || Loc Loss: {reg_loss:.4f} || "
            f"Dir Loss: {dir_loss:.4f} || IoU Loss: {iou_loss:.4f} || Depth Loss: {depth_loss:.4f} || "
            f"Pyramid Loss: {pyramid_loss:.4f} || CU Loss: {cu_loss:.4f}"
        )

        if not writer is None:
            writer.add_scalar("Regression_loss" + suffix, reg_loss, epoch * batch_len + batch_id)
            writer.add_scalar("Confidence_loss" + suffix, cls_loss, epoch * batch_len + batch_id)
            writer.add_scalar("Dir_loss" + suffix, dir_loss, epoch * batch_len + batch_id)
            writer.add_scalar("Iou_loss" + suffix, iou_loss, epoch * batch_len + batch_id)
            writer.add_scalar("Depth_loss" + suffix, depth_loss, epoch * batch_len + batch_id)
            writer.add_scalar("Pyramid_loss" + suffix, pyramid_loss, epoch * batch_len + batch_id)
