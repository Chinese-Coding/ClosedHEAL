import argparse
from typing import Union, List

import PIL
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from opencood.diffusion.Converter import Converter
from opencood.diffusion.StableDiffusionDataset import StableDiffusionDataset
from opencood.diffusion.controlnet.diffusion_feature.capture import Capture
from opencood.diffusion.controlnet.diffusion_feature.dpt_processor import DPTProcessor
from opencood.diffusion.controlnet.diffusion_feature.img_processor import ImageProcessor
from opencood.utils.logger import get_logger

logger = get_logger()

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--root_dir", type=str, default="/datasets/OPV2V/train")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=30)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    augmentations = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
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

    dataset = StableDiffusionDataset(args.root_dir)
    dataset.reinitialize()
    dataset.SetTransform(transform_images)
    # TODO: 真运行的时候记得修改这个 num_workers
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, collate_fn=dataset.collate_fn)

    # Image 部分
    logger.important("加载 Img 部分模型")
    img_capture = Capture(device=torch.device("cuda:0"))
    img_processor = ImageProcessor(img_capture)
    img_optimizer = torch.optim.AdamW(
        img_processor.capturer.model.parameters(), lr=1e-4, betas=(0.95, 0.999), weight_decay=1e-6, eps=1e-08
    )
    img_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(img_optimizer, args.epoch, eta_min=1e-6)

    # lidar 部分
    logger.important("加载 Dpt 部分模型")
    dpt_capture = Capture(device=torch.device("cuda:1"))
    projector = Converter()
    dpt_processor = DPTProcessor(dpt_capture)
    dpt_optimizer = torch.optim.AdamW(
        dpt_processor.capturer.model.parameters(), lr=1e-4, betas=(0.95, 0.999), weight_decay=1e-6, eps=1e-08
    )
    dpt_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dpt_optimizer, args.epoch, eta_min=1e-6)

    logger.important("开始训练")
    for i in range(args.epoch):
        progress_bar = tqdm(total=len(data_loader))
        progress_bar.set_description(f"Epoch {i}")
        for batch in data_loader:
            # 输入到模型的图片总数是: batch_size * 4
            camera = batch["camera"]
            batch_size, num_cameras, channels, height, width = camera.shape  # num_cameras 恒定为 4, channels 恒定为 3
            print(f"camera 的 shape: {camera.shape}")
            img = camera.view(-1, channels, height, width)
            img_noise, img_pred_noise = img_processor(img.to(img_capture.device))

            # lidar 部分
            lidar = batch["lidar"]
            assert isinstance(lidar, list) and len(lidar) == 1
            lidar = lidar[0]
            # TODO: 咱们的点云是 4 维的, 参考项目的是三维的, 没法直接用啊
            # TODO: 投影方式需要改变一下
            dpt = projector.proj_pc2dpt(
                lidar, extrinsic=np.eye(4), intrinsic=np.eye(3), h=height, w=width
            )
            print(type(dpt))
            _, dpt = dpt_processor.process_given_dpt(dpt)
            dpt = (dpt * 1000.0).astype(np.uint16) # 别问, 问就是拿过来的. (最开始他是扩大 1000 倍之后存入磁盘中, 然后需要的时候再读取出来)
            dpt = dpt_processor.control_input(dpt)
            dpt_noise, dpt_pred_noise = dpt_processor(dpt)

            img_loss = torch.nn.functional.mse_loss(img_noise, img_pred_noise)
            dpt_loss = torch.nn.functional.mse_loss(dpt_noise, dpt_pred_noise)

            img_optimizer.zero_grad()
            img_loss.backward()
            img_optimizer.step()
            img_lr_scheduler.step()

            dpt_optimizer.zero_grad()
            dpt_loss.backward()
            dpt_optimizer.step()
            dpt_lr_scheduler.step()

            progress_bar.update(1)
            logs = {
                "img_loss": img_loss.detach().item(), "img_lr": img_lr_scheduler.get_last_lr()[0],
                "dpt_loss": dpt_loss.detach().item(), "dpt_lr": dpt_lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
        progress_bar.close()
