import argparse
import gc
import os
import platform

import torch
from diffusers import StableDiffusionPipeline
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import BuildDataset
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


def PrintTensorInfo(tensor, name):
    info = f"Tensor Name: {name}, Device: {tensor.device}, Shape: {tensor.shape}, Dtype: {tensor.dtype}"
    print(info)


def main():
    torch.set_float32_matmul_precision("medium")

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
    model_id = "stabilityai/stable-diffusion-2-1"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, ignore_mismatched_sizes=True)
    unet = pipeline.unet
    unet.train()
    vae = pipeline.vae
    scheduler = pipeline.scheduler
    scheduler.set_timesteps(100)
    null_prompt_embeds, _ = pipeline.encode_prompt(
        prompt="", device="cpu", num_images_per_prompt=1, do_classifier_free_guidance=False
    )
    opt = torch.optim.Adam(unet.parameters(), lr=1e-6)
    gc.collect()

    device = torch.device("cuda")
    unet.to(device)
    vae.to(device)
    null_prompt_embeds = null_prompt_embeds.cuda()

    for batch in tqdm(train_loader):
        for one_image in batch:
            x = torch.unsqueeze(one_image, dim=0)
            x = x.to(device)
            latents = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
            noise = scheduler.init_noise_sigma * torch.ones_like(latents)
            for t in scheduler.timesteps:
                noisy_latents = scheduler.add_noise(latents, noise, t)
                pred_noise = unet(noisy_latents, t, null_prompt_embeds).sample
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(pred_noise, noise)
                with torch.autograd.set_detect_anomaly(True):
                    loss.backward(retain_graph=True)
                opt.step()
                noise = scheduler.step(pred_noise, t, noise).prev_sample


if __name__ == "__main__":
    main()
