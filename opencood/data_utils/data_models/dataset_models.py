from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict

"""
PF: path for
"""


class PFTimestampData(BaseModel):
    yaml: str
    lidar: str
    cameras: List[Path]
    depths: List[Path]
    modality_name: Optional[str] = None

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


class CAVData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ego: bool
    params: Dict
    camera_data: List[Image]
    depth_data: List[Image]

    lidar_np: Optional[np.ndarray] = None
    modality_name: Optional[str] = None
    file_extension: Optional[Dict] = None

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
