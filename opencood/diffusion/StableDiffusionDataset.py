import os
from pathlib import Path
from typing import List, Dict, Union

import PIL
import numpy as np
import torch
from PIL import Image
from pydantic import ConfigDict, BaseModel, SkipValidation
from torch import Tensor
from torch.utils.data import Dataset

from opencood.hypes_yaml.yaml_utils import LoadYAML, LoadYAMLFromStr
from opencood.utils.camera_utils import LoadCameraData
from opencood.utils.logger import get_logger
from opencood.utils.pcd_utils import pcd_to_np

logger = get_logger()


class CAVData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    camera_data: SkipValidation[List[Image.Image]]
    lidar_np: np.ndarray[np.float64]


class PFTimestampData(BaseModel):
    lidar: str
    cameras: List[Path]


def _GetTimestampDataPath(cavPath: Path, timestamp: str):
    """
    获取某一时间戳下数据的路径

    :param cavPath 汽车所在路径
    :param timestamp 时间戳
    """
    yaml_file, lidar_file = cavPath / f"{timestamp}.yaml", os.path.join(cavPath, f"{timestamp}.pcd")
    camera_files = [cavPath / f"{timestamp}_camera{i}.png" for i in range(4)]
    cavPath = Path(str(cavPath).replace("OPV2V", "OPV2V_Hetero"))
    depth_files = [cavPath / f"{timestamp}_depth{i}.png" for i in range(4)]

    return yaml_file, lidar_file, camera_files, depth_files


def _LoadParams(yamlFile):
    """Load params from YAML (同时将嵌套字典中的列表数据递归转换为 np.ndarray)"""
    if isinstance(yamlFile, Path):
        params = LoadYAML(yamlFile)
    elif isinstance(yamlFile, str):
        params = LoadYAMLFromStr(yamlFile)

    def _ConvertToArray(data):
        match data:
            case dict():
                return {k: _ConvertToArray(v) for k, v in data.items()}
            case list():
                return np.array(data)
            case _:
                return data

    return _ConvertToArray(params)  # 对 params 进行递归处理


def _ExtractValues(d: Dict):
    result = []
    for value in d.values():
        if isinstance(value, dict):
            result.extend(_ExtractValues(value))  # 如果值是字典，则递归处理
        else:
            result.append(value)  # 否则直接添加值
    return result


class StableDiffusionDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        logger.important(f"从 {root_dir} 中加载数据")
        self.scenario_folders: List[Path] = sorted(folder for folder in Path(root_dir).iterdir() if folder.is_dir())

        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database: List[Dict[str, Dict[str, PFTimestampData]]] = []
        self.flattened_database = []

    def SetTransform(self, transform):
        self.transform = transform

    def _Load4DataPaths(self, cav_path: Path):
        outputs = {}
        yaml_files: List[Path] = sorted(file for file in cav_path.glob("*.yaml") if "additional" not in file.stem)
        # this timestamp is not ready
        yaml_files = [
            x for x in yaml_files
            if not (("2021_08_20_21_10_24" in (path_str := str(x)) and "000265" in path_str) or "2021_09_09_13_20_58" in path_str) # fmt: skip
        ]
        timestamps = [file.stem for file in yaml_files]  # 来自GPT: 把提取 timestamp 函数删掉了 (一行代码完事)

        for timestamp in timestamps:
            # 将加载数据路径的函数, 移到了一个单独的函数中 (如果因为后面的代码还需要 `lidar_file` 我一定会让 `_GetTimestampDataPath` 函数返回一个字典)
            yaml_file, lidar_file, camera_files, depth_files = _GetTimestampDataPath(cav_path, timestamp)
            pfTimestampData = PFTimestampData(lidar=lidar_file, cameras=camera_files)
            outputs[timestamp] = pfTimestampData
        return outputs, len(timestamps)

    def reinitialize(self):
        # 每次初始化的时候记得清空之前存储的东西 (如果是第一次初始化可能不需要, 但是为了统一写法就不做判断了)
        self.scenario_database.clear()
        # 定义一个新变量用于存储加载数据的方法, 这样写能缩短代码的长度, 其实也

        # loop over all scenarios
        for i, scenario_folder in enumerate(self.scenario_folders):
            self.scenario_database.append({})

            # at least 1 cav should show up
            # 用三元运算符来简化判断 (使用 sample 函数代替原先的 shuffle 函数, 因为sample函数有返回值写起来比较统一, 不知道应不影响性能)
            cav_list: List[str] = [cav.name for cav in scenario_folder.iterdir() if cav.is_dir()]

            # loop over all CAV data
            for j, cav_id in enumerate(cav_list):
                # save all yaml files to the dictionary
                cav_path = scenario_folder / cav_id
                outputs, timestampsLen = self._Load4DataPaths(cav_path)
                self.scenario_database[i][cav_id] = outputs
        for scenario in self.scenario_database:
            self.flattened_database.extend(_ExtractValues(scenario))

        logger.important(f"数据总长度: {len(self.flattened_database)}")

    def __getitem__(self, idx):
        """
        Given the index, return the corresponding data.

        :param idx: Index given by dataloader
        :return: The dictionary contains loaded yaml params and lidar data for each cav.
        """
        pathes = self.flattened_database[idx]
        return CAVData(camera_data=LoadCameraData(pathes.cameras), lidar_np=pcd_to_np(pathes.lidar))

    def __len__(self):
        return len(self.flattened_database)

    def collate_fn(self, batches: List[CAVData]) -> Dict[str, Tensor]:
        camera_data, lidar_np = [], []  # type: List[Tensor], List[Tensor]
        for batch in batches:
            camera_data.append(torch.stack(self.transform(batch.camera_data)))
            lidar_np.append(torch.tensor(batch.lidar_np))
        return {
            # camera shape: (batch, 4, 3, W, H), 这个 4 是每个车有四个相机
            "camera": torch.stack(camera_data),
            "lidar": torch.stack(lidar_np),
        }


class OPV2VDiffusionDataset(Dataset):
    def __init__(self, root_dir: str):
        print(f"从 {root_dir} 中加载数据")
        self.img_path = sorted(file for file in Path(root_dir).rglob("*.png"))
        print(f"共加载 {len(self.img_path)} 张图片")

    def SetTransform(self, transform):
        self.transform = transform

    def reinitialize(self):
        pass

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx) -> torch.Tensor:
        return self.transform(Image.open(self.img_path[idx]).copy())

    def collate_fn(self, imgs: List[Image]) -> torch.Tensor:
        return torch.stack(imgs)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision import transforms

    resolution = 512

    augmentations = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def transform_images(examples: Union[List[PIL.Image.Image], PIL.Image.Image]) -> Union[List[torch.Tensor], torch.Tensor]:
        if isinstance(examples, list):
            images = [augmentations(image.convert("RGB")) for image in examples]
        elif isinstance(examples, PIL.Image.Image):
            images = augmentations(examples.convert("RGB"))
        return images

    dataset = StableDiffusionDataset("/datasets/OPV2V/train")
    dataset.reinitialize()
    dataset.SetTransform(transform_images)
    dataLoader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)
    for batch in dataLoader:
        print(f"batch 的类型为: {type(batch)}")

        break
