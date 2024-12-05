import argparse
import os
import platform

import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import BuildDataset
from opencood.models.diffusion.opv2v_diffusion_model import OPV2VDiffusionModel
from opencood.utils.logger import get_logger

logger = get_logger()


def _PrintSystemInfo():
    print(f"""
    操作系统以及版本: {platform.system()} {platform.version()}
    计算机名称与用户名: {platform.node()} {os.getlogin()}
    Python 与 pytorch 版本: {platform.python_version()} {torch.__version__}
    """)


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True, help="data generation yaml file needed ")
    parser.add_argument("--model_dir", default="", help="Continued training path")
    parser.add_argument("--fusion_method", "-f", default="intermediate", help="passed to inference.")
    parser.add_argument("--gpus", default=1, help="使用的 GPU 个数")
    opt = parser.parse_args()
    return opt


def main():
    _PrintSystemInfo()
    os.system("python opencood/utils/setup.py build_ext --inplace")  # 每次执行前都先编译一下, 以免修改了忘记编译了
    opt = train_parser()
    hypes = yaml_utils.LoadYAML(opt.hypes_yaml, opt)

    logger.important("Dataset Building")
    trainDataset, evalDataset = BuildDataset(hypes), BuildDataset(hypes, train=False)

    train_loader = DataLoader(
        trainDataset,
        batch_size=hypes["train_params"]["batch_size"],
        num_workers=2,
        collate_fn=trainDataset.collate_batch_train,  # WARNING: 如果想要全部数据进行训练需要修改这里
        shuffle=True,
        pin_memory=True,  # 这里先改成 False, 先跑起来再说
        drop_last=True,
        prefetch_factor=2,
    )
    # val_loader = DataLoader(
    #     evalDataset,
    #     batch_size=hypes["train_params"]["batch_size"],
    #     num_workers=2,
    #     collate_fn=trainDataset.collate_batch_train,  # WARNING: 如果想要全部数据进行训练需要修改这里
    #     shuffle=True,
    #     pin_memory=True,  # 这里先改成 False, 先跑起来再说
    #     drop_last=True,
    #     prefetch_factor=2,
    # )

    logger.important("数据集加载完毕, 开始创建模型")
    model = OPV2VDiffusionModel(train_loader, hypes)
    trainer = L.Trainer(
        accelerator="gpu",  # 指定使用GPU
        devices=-1,
        max_epochs=hypes["train_params"]["epoches"],
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        val_check_interval=1,
        precision=16,
        strategy="ddp",
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
