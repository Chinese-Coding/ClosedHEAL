import json
import math
import os
import re
from typing import Dict

import yaml

from logger import get_logger

logger = get_logger()

pattern = """^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$"""


def load_yaml(file: str, model_dir=None):
    """
    加载原始配置文件, 并根据原始配置文件的要求对配置文件的内容进行修改
    :param file:
    :param model_dir:
    :return:
    """
    if model_dir:
        # WARNING: 如果从 `model_dir` 文件加载时就是从 `config.yaml` 中加载, 所以一定要注意命名问题.
        file = os.path.join(model_dir, "config.yaml")
    try:
        with open(file, "r") as stream:
            loader = yaml.Loader
            loader.add_implicit_resolver("tag:yaml.org,2002:float", re.compile(pattern, re.X), list("-+0123456789."))
            configs = yaml.load(stream, Loader=loader)
    except FileNotFoundError:
        logger.critical(f"{file} 没有找到")
    except yaml.YAMLError as e:
        logger.critical(f"解析文件时出错: {e}")

    # 处理字符串插值
    if "root_dir" in configs:
        root_dir = configs["root_dir"]
        configs["train_dir"] = configs["train_dir"].replace("${root_dir}", root_dir)
        configs["validate_dir"] = configs["validate_dir"].replace("${root_dir}", root_dir)
        configs["test_dir"] = configs["test_dir"].replace("${root_dir}", root_dir)

    if "yaml_parser" in configs:
        # TODO: 修改这种 `eval` 函数的写法
        configs = eval(configs["yaml_parser"])(configs)
    return configs


def load_general_params(configs: Dict) -> Dict:
    """
    Based on the lidar range and resolution of voxel, calculate the anchor box and target resolution.

    :param configs: Original loaded parameter dictionary.
    :return: Modified parameter dictionary with new attribute.
    """
    cav_lidar_range = configs["preprocess"]["cav_lidar_range"]
    voxel_size = configs["preprocess"]["args"]["voxel_size"]
    anchor_args = configs["postprocess"]["anchor_args"]

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args["vw"] = vw
    anchor_args["vh"] = vh
    anchor_args["vd"] = vd

    # W is image width, but along with x axis in lidar coordinate
    anchor_args["W"] = math.ceil((cav_lidar_range[3] - cav_lidar_range[0]) / vw)
    anchor_args["H"] = math.ceil((cav_lidar_range[4] - cav_lidar_range[1]) / vh)  # H is image height
    anchor_args["D"] = math.ceil((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    configs["postprocess"].update({"anchor_args": anchor_args})

    return configs


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data
