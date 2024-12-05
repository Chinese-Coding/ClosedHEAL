# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import platform
import statistics
import time

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import BuildDataset
from opencood.tools import train_utils
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
    opt = parser.parse_args()
    return opt


def _BuildModules(hypes):
    model = train_utils.create_model(hypes)
    criterion = train_utils.create_loss(hypes)  # define the loss
    optimizer = train_utils.setup_optimizer(hypes, model)  # optimizer setup
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer)
    return model, criterion, optimizer, scheduler


def _LoadModules(model_dir, hypes, model, optimizer):
    if model_dir:
        saved_path = model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
        logger.important(f"resume from {init_epoch} epoch.")
    else:
        saved_path = train_utils.setup_train(hypes)
        init_epoch, lowest_val_epoch = 0, -1

    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch)
    return saved_path, init_epoch, lowest_val_epoch, model, scheduler


def _TrainOneEpoch(dataloader, device, epoch, writer, model, optimizer, criterion, single_weight=1):
    length = len(dataloader)
    for i, batch_data in enumerate(dataloader):
        if batch_data is None or batch_data["ego"]["object_bbx_mask"].sum() == 0:
            continue
        model.zero_grad()
        optimizer.zero_grad()
        batch_data = train_utils.to_device(batch_data, device)
        batch_data["ego"]["epoch"] = epoch
        output_dict = model(batch_data["ego"])

        final_loss = criterion(output_dict, batch_data["ego"]["label_dict"])
        criterion.logging(epoch, i, length, writer)
        # 这个东西是在 `supervise_signle_flag == true` 时才会用到, 因为配置文件里默认为 true, 所以就写在这里了
        # 不知道 `supervise_signle_flag` 有什么用, 怀疑和 Loss 一直不下降有关系
        final_loss += criterion(output_dict, batch_data["ego"]["label_dict_single"], suffix="_single") * single_weight
        criterion.logging(epoch, i, length, writer, suffix="_single")
        final_loss.backward()
        optimizer.step()


def _EvalOneEpoch(dataloader, device, epoch, writer, model, criterion):
    valid_ave_loss = []
    model.eval()  # 将 eval 模式放到循环外
    with torch.no_grad():  # 全局使用 no_grad(), 提高效率
        for i, batch_data in enumerate(dataloader):
            if batch_data is None:
                continue
            batch_data = train_utils.to_device(batch_data, device)
            batch_data["ego"]["epoch"] = epoch
            output_dict = model(batch_data["ego"])

            final_loss = criterion(output_dict, batch_data["ego"]["label_dict"])
            print(f"val loss {final_loss:.3f}")
            valid_ave_loss.append(final_loss.item())
    valid_ave_loss = statistics.mean(valid_ave_loss)
    print(f"At epoch {epoch}, the validation loss is {valid_ave_loss:.6f}")
    writer.add_scalar("Validate_loss", valid_ave_loss, epoch)
    return valid_ave_loss


def _SaveModel(lowest_val_loss, valid_ave_loss, lowest_val_epoch, epoch, saved_path, model):
    if valid_ave_loss < lowest_val_loss:
        lowest_val_loss, best_model_path = valid_ave_loss, os.path.join(saved_path, f"net_epoch_bestval_at{epoch+1}.pth")
        torch.save(model.state_dict(), best_model_path)

        previous_model_path = os.path.join(saved_path, f"net_epoch_bestval_at{lowest_val_epoch}.pth")
        if lowest_val_epoch != -1 and os.path.exists(previous_model_path):
            os.remove(previous_model_path)
        lowest_val_epoch = epoch + 1
    return lowest_val_loss, lowest_val_epoch


def main():
    _PrintSystemInfo()
    os.system("python opencood/utils/setup.py build_ext --inplace")  # 每次执行前都先编译一下, 以免修改了忘记编译了
    opt = train_parser()
    hypes = yaml_utils.LoadYAML(opt.hypes_yaml, opt)

    logger.important("Dataset Building")
    trainDataset, evalDataset = BuildDataset(hypes), BuildDataset(hypes, train=False)

    # trainDataset, evalDataset = Subset(trainDataset, range(0, 100)), Subset(evalDataset, range(0, 100))

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.important("数据集加载完毕, 开始创建模型")
    model, criterion, optimizer, scheduler = _BuildModules(hypes)
    # if we want to train from last checkpoint.
    lowest_val_loss, lowest_val_epoch = 1e5, -1
    saved_path, init_epoch, lowest_val_epoch, model, scheduler = _LoadModules(opt.model_dir, hypes, model, optimizer)
    logger.add(saved_path)  # 获取到新 `save_path` 后新增加一个输出
    model.to(device)

    # record training
    writer = SummaryWriter(saved_path)

    epoches, single_weight = hypes["train_params"]["epoches"], hypes["train_params"].get("single_weight", 1)

    # used to help schedule learning rate
    usingTimeList = []
    logger.important("开始循环")
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        startTime = time.time()

        trainDataset.baseDataset.reinitialize()
        for param_group in optimizer.param_groups:
            print("learning rate %f" % param_group["lr"])
        # the model will be evaluation mode during validation
        model.train()
        try:  # heter_model stage2
            model.model_train_init()
        except:
            print("No model_train_init function")

        _TrainOneEpoch(train_loader, device, epoch, writer, model, optimizer, criterion, single_weight)

        if epoch % hypes["train_params"]["save_freq"] == 0:
            torch.save(model.state_dict(), os.path.join(saved_path, f"net_epoch{epoch+1}.pth"))

        if epoch % hypes["train_params"]["eval_freq"] == 0:
            valid_ave_loss = _EvalOneEpoch(val_loader, device, epoch, writer, model, criterion)
            lowest_val_loss, lowest_val_epoch = _SaveModel(
                lowest_val_loss, valid_ave_loss, lowest_val_epoch, epoch, saved_path, model
            )

        scheduler.step()
        usingTime = time.time() - startTime
        usingTimeList.append(usingTime)
        logger.important(f"{epoch} 训练和评估时长: {usingTime}s")

    logger.important(f"总训练时长: {sum(usingTimeList)}s, 一轮平均耗时: {sum(usingTimeList) / len(usingTimeList)}s")
    logger.important(f"Training Finished, checkpoints saved to {saved_path}")


if __name__ == "__main__":
    main()
