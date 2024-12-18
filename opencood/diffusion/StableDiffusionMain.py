import argparse
from typing import Union, List

import PIL
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from opencood.diffusion.StableDiffusionDataset import StableDiffusionDataset, OPV2VDiffusionDataset
from opencood.diffusion.controlnet.diffusion_feature.capture import Capture
from opencood.diffusion.controlnet.diffusion_feature.image_processor import ImageProcessor


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
    image_capture = Capture(device=torch.device("cuda:0"))
    image_processor = ImageProcessor(image_capture)
    image_optimizer = torch.optim.AdamW(
        image_processor.capturer.model.parameters(), lr=1e-4, betas=(0.95, 0.999), weight_decay=1e-6, eps=1e-08
    )
    image_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(image_optimizer, args.epoch, eta_min=1e-6)

    for i in range(args.epoch):
        progress_bar = tqdm(total=len(data_loader))
        progress_bar.set_description(f"Epoch {i}")
        for batch in data_loader:
            camera = batch["camera"]
            # 输入到模型的图片总数是: batch_size * 4
            # 取 4 张图片里面的一张图片输入就没有问题, 但是一次性输入 4 张图片就有问题了
            batch_size, num_cameras, channels, height, width = camera.shape  # num_cameras 恒定为 4, channels 恒定为 3
            camera = camera.view(-1, channels, height, width)
            noise, pred_noise = image_processor(camera.to(image_capture.device))

            # noise, pred_noise = image_processor(camera[0].unsqueeze(0).to(image_capture.device))
            # noise, pred_noise = image_processor(batch.to(image_capture.device))
            loss = torch.nn.functional.mse_loss(noise, pred_noise)
            image_optimizer.zero_grad()
            loss.backward()
            image_optimizer.step()
            image_lr_scheduler.step()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": image_lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
        progress_bar.close()
