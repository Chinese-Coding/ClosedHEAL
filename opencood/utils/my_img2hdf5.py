# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
import argparse
import time
from multiprocessing import Process
from pathlib import Path
from threading import Thread
from typing import List

import h5py
import numpy as np
from PIL import Image
from loguru import logger
from tqdm import tqdm


def _LoadCameraData(camera_files: List[Path], preLoad=True):
    """
    在 Python 的 PIL 库中，Image.open() 函数并不会立刻把整个图像数据加载到内存中，而是采用“延迟加载”的方式，
    即仅在需要时（例如第一次访问像素数据时）才会实际加载。这种懒加载方式可以节省内存，但在并发访问时可能会导致数据被释放或修改。

    通过 preload=True，代码会在读取图像后立即调用 .copy()，从而将图像数据实际加载到内存中，提高运行速度
    """
    return (
        [Image.open(camera_file).copy() for camera_file in camera_files]
        if preLoad
        else [Image.open(camera_file) for camera_file in camera_files]
    )


def _LoadCameraFiles(cav_path: Path, timestamp: str, name: str):
    return [cav_path / f"{timestamp}_{name}{i}.png" for i in range(4)]


def _LoadDepthFiles(cav_path: Path, timestamp: str, name: str):
    cav_path = Path(str(cav_path).replace("OPV2V", "OPV2V_Hetero"))  # 使用 `str` 会创建一个新对象不会影响原来的路径字符串
    return _LoadCameraFiles(cav_path, timestamp, name)


def _LoadHeteroFiles(cav_path: Path, timestamp: str):
    lidar_files = {"64": cav_path / f"{timestamp}.pcd"}
    cav_path = Path(str(cav_path).replace("OPV2V", "OPV2V_Hetero"))  # 使用 `str` 会创建一个新对象不会影响原来的路径字符串
    lidar_files.update({"32": cav_path / f"{timestamp}_32.pcd", "16": cav_path / f"{timestamp}_16.pcd"})
    depth_files = [cav_path / f"{timestamp}_depth{i}.png" for i in range(4)]
    return lidar_files, depth_files


def _Transform(cav_path: Path, timestamp: str):
    hdf5 = cav_path / f"{timestamp}_imgs.hdf5"
    if hdf5.exists():
        logger.warning(f"已存在 {hdf5} 文件, 跳过")
        return
    camera_files = _LoadCameraFiles(cav_path, timestamp, name="camera")
    depth_files = _LoadDepthFiles(cav_path, timestamp, name="depth")

    camera_data, depth_data = _LoadCameraData(camera_files), _LoadCameraData(depth_files)

    with h5py.File(hdf5, "w") as f:
        for i in range(4):
            f.create_dataset(f"camera{i}", data=camera_data[i])
        for i in range(4):
            f.create_dataset(f"depth{i}", data=depth_data[i])
    # logger.success(f"{hdf5} 创建成功")


def _TransformAll(cav_path: Path, timestamp: str):
    hdf5 = cav_path / f"{timestamp}.hdf5"
    if hdf5.exists():
        logger.warning(f"已存在 {hdf5} 文件, 跳过")
        return

    yaml_file, camera_files = cav_path / f"{timestamp}.yaml", [cav_path / f"{timestamp}_camera{i}.png" for i in range(4)]
    lidar_files, depth_files = _LoadHeteroFiles(cav_path, timestamp)

    camera_data, depth_data = _LoadCameraData(camera_files), _LoadCameraData(depth_files)

    with h5py.File(hdf5, "w") as f:
        with open(yaml_file, "rb") as yf:
            f.create_dataset("yaml_file", data=yf.read(), dtype=h5py.special_dtype(vlen=bytes))

        for k, v in lidar_files.items():

            with open(v, "rb") as lf:
                f.create_dataset(k, data=lf.read(), dtype=h5py.special_dtype(vlen=bytes))

        for i in range(4):
            f.create_dataset(f"camera{i}", data=camera_data[i])
        for i in range(4):
            f.create_dataset(f"depth{i}", data=depth_data[i])
    # logger.success(f"{hdf5} 创建成功")


def _ClearUp(cav_path: Path, timestamp: str):
    hdf5 = cav_path / f"{timestamp}_imgs.hdf5"
    if hdf5.exists():
        hdf5.unlink()
        # logger.success("文件删除成功")


def _ClearUpAll(cav_path: Path, timestamp: str):
    hdf5 = cav_path / f"{timestamp}.hdf5"
    if hdf5.exists():
        hdf5.unlink()


def _Parallel(scenario_folders: List[Path], fun, processId: int):
    logger.success(f"subprocess {processId} 启动!")
    startTime = time.time()
    for scenario_folder in tqdm(scenario_folders, desc=f"Process {processId}", position=processId, leave=False):
        cav_list: List[str] = [cav.name for cav in scenario_folder.iterdir() if cav.is_dir()]
        assert len(cav_list) > 0

        # loop over all CAV data
        for cav_id in cav_list:
            cav_path = scenario_folder / cav_id
            yaml_files: List[Path] = sorted(file for file in cav_path.glob("*.yaml") if "additional" not in file.stem)
            timestamps = [file.stem for file in yaml_files]
            for timestamp in timestamps:
                try:
                    fun(cav_path, timestamp)
                except FileNotFoundError as e:
                    logger.error(f"File not found in process {processId}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error in process {processId}: {e}")
    logger.success(f"subprocess {processId} finished! 消耗时间: {time.time() - startTime}")


def _ThreadVersion(MP_NUM, mp_split):
    """
    多线程版本, 需要没有 GIL 版本的 python 才行
    (没有做过测试, 因为最新的 python 3.13 虽然说没有 GIL 了, 但是很多库并不适配, 想用其他语言的 python 实现则是更不可能的事情了)
    """
    threads = []
    for i in range(MP_NUM):
        t = Thread(target=_Parallel, args=(mp_split[i], _Transform, i))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


def _ProcessVersion(MP_NUM, mp_split, fun):
    processes = [Process(target=_Parallel, args=(mp_split[i], fun, i)) for i in range(MP_NUM)]
    # 奇怪必须用列表推导式子来干这件事
    [process.start() for process in processes]
    [process.join() for process in processes]


def _Parser():
    parser = argparse.ArgumentParser(description="数据集到 HDF5 数据类型转换")
    parser.add_argument("--processNumber", default=8)
    parser.add_argument("--function", default="ClearUPAll")
    return parser.parse_args()


if __name__ == "__main__":
    startTime = time.time()
    opt = _Parser()
    funs = {"ClearUp": _ClearUp, "ClearUpAll": _ClearUpAll, "Transform": _Transform, "TransformAll": _TransformAll}
    MP_NUM = opt.processNumber
    fun = funs[opt.function]
    rootDir = Path("~/Desktop").expanduser()
    split_folders = [rootDir / f"dataset/OPV2V/{split}" for split in ["train", "validate", "test"]]
    scenario_folders = []
    logger.success(f"需要替换的目录: {split_folders}")

    scenario_folders = [
        subfolder
        for folder in split_folders
        if folder.exists() and folder.is_dir()
        for subfolder in sorted(folder.iterdir())
        if subfolder.is_dir()
    ]
    mp_split = np.array_split(scenario_folders, MP_NUM)
    mp_split = [x.tolist() for x in mp_split]

    _ProcessVersion(MP_NUM, mp_split, fun)
    logger.success(f"总消耗时间: {time.time() - startTime}")
