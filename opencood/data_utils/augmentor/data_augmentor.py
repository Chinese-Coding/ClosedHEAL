# -*- coding: utf-8 -*-
"""
Class for data augmentation
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from functools import partial

from opencood.data_utils.augmentor import augment_utils


def _RandomWorldScaling(data_dict=None, config=None):
    gt_boxes, gt_mask, points = data_dict["object_bbx_center"], data_dict["object_bbx_mask"], data_dict["lidar_np"]
    gt_boxes_valid = gt_boxes[gt_mask == 1]

    gt_boxes_valid, points = augment_utils.global_scaling(gt_boxes_valid, points, config["WORLD_SCALE_RANGE"])
    gt_boxes[: gt_boxes_valid.shape[0], :] = gt_boxes_valid

    data_dict["object_bbx_center"] = gt_boxes
    data_dict["object_bbx_mask"] = gt_mask
    data_dict["lidar_np"] = points

    return data_dict


def _RandomWorldRotation(data_dict=None, config=None):
    rot_range = config["WORLD_ROT_ANGLE"]
    if not isinstance(rot_range, list):
        rot_range = [-rot_range, rot_range]

    gt_boxes, gt_mask, points = data_dict["object_bbx_center"], data_dict["object_bbx_mask"], data_dict["lidar_np"]
    gt_boxes_valid = gt_boxes[gt_mask == 1]
    gt_boxes_valid, points = augment_utils.global_rotation(gt_boxes_valid, points, rot_range=rot_range)
    gt_boxes[: gt_boxes_valid.shape[0], :] = gt_boxes_valid

    data_dict["object_bbx_center"] = gt_boxes
    data_dict["object_bbx_mask"] = gt_mask
    data_dict["lidar_np"] = points

    return data_dict


def _RandomWorldFlip(data_dict=None, config=None):
    gt_boxes, gt_mask, points = data_dict["object_bbx_center"], data_dict["object_bbx_mask"], data_dict["lidar_np"]
    gt_boxes_valid = gt_boxes[gt_mask == 1]

    for cur_axis in config["ALONG_AXIS_LIST"]:
        if cur_axis not in ["x", "y"]:
            raise ValueError(f'Axis must be "x" or "y", but you input is {cur_axis}')
        gt_boxes_valid, points = getattr(augment_utils, f"random_flip_along_{cur_axis}")(gt_boxes_valid, points)

    gt_boxes[: gt_boxes_valid.shape[0], :] = gt_boxes_valid

    data_dict["object_bbx_center"] = gt_boxes
    data_dict["object_bbx_mask"] = gt_mask
    data_dict["lidar_np"] = points

    return data_dict


class DataAugmentor:
    """
    Data Augmentor.

    Parameters
    ----------
    augment_config : list
        A list of augmentation configuration.

    Attributes
    ----------
    data_augmentor_queue : list
        The list of data augmented functions.
    """

    def __init__(self, augment_config, train=True):
        self.data_augmentor_queue, self.train = [], train

        # Mapping of augmentor name to methods
        augmentorFuns = {
            "random_world_flip": _RandomWorldFlip,
            "random_world_rotation": _RandomWorldRotation,
            "random_world_scaling": _RandomWorldScaling,
        }

        for cur_cfg in augment_config:
            augmentor_name = cur_cfg["NAME"]
            if augmentor_name in augmentorFuns:
                self.data_augmentor_queue.append(partial(augmentorFuns[augmentor_name], config=cur_cfg))
            else:
                raise ValueError(f"Unknown augmentor: {augmentor_name}")

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        if self.train:
            for cur_augmentor in self.data_augmentor_queue:
                data_dict = cur_augmentor(data_dict=data_dict)
        return data_dict
