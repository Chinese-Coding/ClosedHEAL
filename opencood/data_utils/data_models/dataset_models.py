from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

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
