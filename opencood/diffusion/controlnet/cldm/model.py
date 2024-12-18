import os

import torch
from omegaconf import OmegaConf

from opencood.diffusion.controlnet.ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get("state_dict", d)


def load_state_dict(ckpt_path, location="cpu"):
    _, extension = os.path.splitext(ckpt_path)
    # 如果扩展名是 `.safetensors`, 则使用 `safetensors.torch.load_file()` 函数加载文件, 这是一种常见的 PyTorch 的 tensor 文件格式.
    if extension.lower() == ".safetensors":
        import safetensors.torch

        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location), weights_only=False))
    state_dict = get_state_dict(state_dict)
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    return model
