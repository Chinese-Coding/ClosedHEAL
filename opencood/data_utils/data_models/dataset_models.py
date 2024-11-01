from pathlib import Path
from typing import List, Optional, Dict, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, SkipValidation

"""
PF: path for
"""


class PFTimestampData(BaseModel):
    yaml: Path
    lidar: str
    cameras: List[Path]
    depths: List[Path]
    modality_name: Optional[str] = None
    file_extensions: Dict[str, str] = Field(default_factory=dict)

    def __getitem__(self, key):
        if key in self.__dict__:
            return getattr(self, key)
        else:
            raise KeyError(f"Key '{key}' not found in {self.__class__.__name__}")

    def __setitem__(self, key, value):
        if key in self.__dict__:
            setattr(self, key, value)
        else:
            raise KeyError(f"Key '{key}' not found in {self.__class__.__name__}")


class HDF5Data(BaseModel):
    filePath: Path
    # lidarLines: Literal["16", "32", "64"] = "64" # 因为 open3d 不支持从内存中读取 pcd 形式的点云, 所以暂时放弃这个字段
    lidar: str
    modality_name: Optional[str] = None
    file_extensions: Dict[str, str] = Field(default_factory=dict)


class CAVData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ego: bool
    params: Dict
    camera_data: SkipValidation[List[Image.Image]]
    depth_data: SkipValidation[List[Image.Image]]

    lidar_np: Optional[np.ndarray[np.float64]] = None
    modality_name: Optional[str] = None
    file_extensions: Dict = Field(default_factory=dict)

    def __getitem__(self, key):
        if key in self.__dict__:
            return getattr(self, key)
        else:
            raise KeyError(f"Key '{key}' not found in {self.__class__.__name__}")

    def __setitem__(self, key, value):
        if key in self.__dict__:
            setattr(self, key, value)
        else:
            raise KeyError(f"Key '{key}' not found in {self.__class__.__name__}")
