from pathlib import Path
from typing import List

import torch.utils.data.dataset

from PIL import Image


def _load_camera_data(cav_path: Path):
    outputs = {}
    yaml_files: List[Path] = sorted(file for file in cav_path.glob("*.yaml") if "additional" not in file.stem)
    timestamps = [file.stem for file in yaml_files]
    for timestamp in timestamps:
        outputs[timestamp] = [cav_path / f"{timestamp}_camera{i}.png" for i in range(4)]
    return outputs


class OPV2VDiffusionDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, root_dir: str):
        print(f"从 {root_dir} 中加载数据")
        self.img_path = sorted(file for file in Path(root_dir).rglob("*.png"))
        print(f"共加载 {len(self.img_path)} 张图片")

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx) -> torch.Tensor:
        return self.transform(Image.open(self.img_path[idx]).copy())

    def collate_batch(self, imgs: List[Image]) -> torch.Tensor:
        return torch.stack(imgs)
