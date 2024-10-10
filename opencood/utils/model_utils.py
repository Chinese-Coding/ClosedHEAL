# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

from collections import OrderedDict

import torch
import torch.nn as nn


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()


def unfix_bn(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.train()


def _CheckBatchNormMode(module: torch.nn.Module, is_training: bool) -> bool:
    return any(
        m.training == is_training
        for m in module.modules()
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))
    )


def has_trainable_params(module: torch.nn.Module) -> bool:
    return any(p.requires_grad for p in module.parameters()) or _CheckBatchNormMode(module, is_training=True)


def has_untrainable_params(module: torch.nn.Module) -> bool:
    return any(not p.requires_grad for p in module.parameters()) or _CheckBatchNormMode(module, is_training=False)


def check_trainable_module(model):
    appeared_module_set = set()
    has_trainable_list, has_untrainable_list = [], []

    for name, module in model.named_modules():
        if name == "" or any(name.startswith(mod_name) for mod_name in appeared_module_set):
            continue

        appeared_module_set.add(name)

        # 只遍历一次参数，记录可训练和不可训练的模块
        trainable_found = has_trainable_params(module)
        untrainable_found = has_untrainable_params(module)

        if trainable_found:
            has_trainable_list.append(name)
        if untrainable_found:
            has_untrainable_list.append(name)

    print("=========Those modules have trainable component=========")
    print("\n".join(has_trainable_list), end="\n\n")
    print("=========Those modules have untrainable component=========")
    print("\n".join(has_untrainable_list), end="\n\n")


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=0.1)
        if hasattr(m.bias, "data"):
            nn.init.constant_(m.bias.data, 0)

    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, gain=0.1)
        # if hasattr(m, 'bias'):
        #     nn.init.constant_(m.bias, 0)

    # elif isinstance(m, nn.BatchNorm2d):
    #     nn.init.xavier_normal_(m.weight, gain=0.05)
    #     nn.init.constant_(m.bias, 0)


def rename_model_dict_keys(model_dict_path, rename_dict):
    """
    Args:
        model_dict_path : str
            path to the model checkpoints
        rename_dict : dict
            key: old name
            value: new name

    Usage:
        Case 1: remove model parameters
            rename_dict = {"camera_encoder.*": "",
                            "camera_backbone.*": "",
                            "shrink_camera.*": "",
                            "cls_head_camera.*": "",
                            "reg_head_camera.*": "",
                            "dir_head_camera.*": "",}
            if the value is "", then the key will be removed from the model dict

        Case 2: rename model parameters' keys
            rename_dict = {"camencode.*": "camera_encoder.camencode.*",
                           "bevencode.*": "camera_encoder.bevencode.*",
                           "head.cls_head.*": "cls_head_camera.*",
                           "head.reg_head.*": "reg_head_camera.*",
                           "head.dir_head.*": "dir_head_camera.*",
                           "shrink_conv.*": "shrink_camera.*"}
            if the value is not "", then the key will be renamed to the value. * is supported to match multiple keys

    """
    pretrained_dict = torch.load(model_dict_path)
    torch.save(pretrained_dict, model_dict_path.replace(".pth", "_before_rename.pth"))
    # 1. filter out unnecessary keys
    for oldname, newname in rename_dict.items():
        if oldname.endswith("*"):
            _oldnames = list(pretrained_dict.keys())
            _oldnames = [x for x in _oldnames if x.startswith(oldname[:-1])]
            for _oldname in _oldnames:
                if newname != "":
                    _newname = _oldname.replace(oldname[:-1], newname[:-1])
                    pretrained_dict[_newname] = pretrained_dict[_oldname]
                pretrained_dict.pop(_oldname)
        else:
            if newname != "":
                pretrained_dict[newname] = pretrained_dict[oldname]
            pretrained_dict.pop(oldname)
    torch.save(pretrained_dict, model_dict_path)


def compose_model(model1, keyname1, model2, keyname2, output_model):
    pretrained_dict1 = torch.load(model1)
    pretrained_dict2 = torch.load(model2)

    new_dict = OrderedDict()
    for keyname in keyname1:
        if keyname.endswith("*"):
            _oldnames = list(pretrained_dict1.keys())
            _oldnames = [x for x in _oldnames if x.startswith(keyname[:-1])]
            for _oldname in _oldnames:
                new_dict[_oldname] = pretrained_dict1[_oldname]

    for keyname in keyname2:
        if keyname.endswith("*"):
            _oldnames = list(pretrained_dict2.keys())
            _oldnames = [x for x in _oldnames if x.startswith(keyname[:-1])]
            for _oldname in _oldnames:
                new_dict[_oldname] = pretrained_dict2[_oldname]

    torch.save(new_dict, output_model)


if __name__ == "__main__":
    # exemplar usage 2: rename model parameters' keys!
    dict_path = "/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs/v2xset_heter_late_fusion/net_epoch_bestval_at28.pth"
    rename_dict = {
        "camencode.*": "camera_encoder.camencode.*",
        "bevencode.*": "camera_encoder.bevencode.*",
        "head.cls_head.*": "cls_head_camera.*",
        "head.reg_head.*": "reg_head_camera.*",
        "head.dir_head.*": "dir_head_camera.*",
        "shrink_conv.*": "shrink_camera.*",
    }
    rename_model_dict_keys(dict_path, rename_dict)
