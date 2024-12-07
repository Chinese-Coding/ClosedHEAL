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
    val_loader = DataLoader(
        evalDataset,
        batch_size=hypes["train_params"]["batch_size"],
        num_workers=2,
        collate_fn=trainDataset.collate_batch_train,  # WARNING: 如果想要全部数据进行训练需要修改这里
        shuffle=True,
        pin_memory=True,  # 这里先改成 False, 先跑起来再说
        drop_last=True,
        prefetch_factor=2,
    )

    logger.important("数据集加载完毕, 开始创建模型")
    # 当您使用具有 Tensor Cores 的 NVIDIA GPU（例如 RTX 4090）时，PyTorch 提示您可以使用 torch.set_float32_matmul_precision('medium' | 'high') 来充分利用这些硬件特性。
    # 默认情况下，PyTorch 的矩阵乘法使用标准的 FP32 精度，但这并不能充分发挥 Tensor Cores 的高效特性。
    # 设置 torch.set_float32_matmul_precision('medium') 或 torch.set_float32_matmul_precision('high') 可以让 PyTorch 使用 TF32 或混合精度的方式在计算矩阵乘法时进行一定程度的精度-性能折中，从而获得更好的训练速度和吞吐量。
    # 简而言之，您的 GPU 支持更高效的矩阵运算模式，通过这条设置可以在计算图开始前添加类似以下代码，从而提升训练的速度（可能会有非常轻微的精度损失）
    torch.set_float32_matmul_precision("medium")
    # 尝试在训练过程中加入torch.autograd.detect_anomaly(True) 检查梯度传播过程
    torch.autograd.detect_anomaly(True)
    model = OPV2VDiffusionModel(hypes)
    trainer = L.Trainer(
        accelerator="gpu",  # 指定使用GPU
        devices=[0, 1],  # 使用全部的 GPU
        max_epochs=hypes["train_params"]["epoches"],
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        val_check_interval=1,
        precision="16-mixed",
        strategy="ddp_find_unused_parameters_true",
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
