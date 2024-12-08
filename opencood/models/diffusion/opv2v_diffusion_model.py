import gc

import pytorch_lightning as L
import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler, StableDiffusionPipeline
from torch import nn


class OPV2VDiffusionModel(nn.Module):
    def __init__(self, optimizer_hype: dict, model_id="stabilityai/stable-diffusion-2-1"):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", ignore_mismatched_sizes=True)
        self.unet.train()  # 使用 unet 的训练模式
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", use_safetensors=True)
        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipeline = StableDiffusionPipeline.from_pretrained(model_id, vae=self.vae, unet=self.unet, scheduler=self.scheduler)
        self.null_prompt_embeds, _ = pipeline.encode_prompt(
            prompt="", device="cpu", num_images_per_prompt=1, do_classifier_free_guidance=False
        )
        gc.collect()
        self.optimizer = self._BuildOptimizer(optimizer_hype)
        # Important: This property activates manual optimization.

    def _BuildOptimizer(self, hypes):
        method_dict = hypes["optimizer"]
        optimizer_method = getattr(torch.optim, method_dict["core_method"], None)
        if not optimizer_method:
            raise ValueError("{} is not supported".format(method_dict["name"]))
        if "args" in method_dict:
            return optimizer_method(self.unet.parameters(), lr=method_dict["lr"], **method_dict["args"])
        else:
            return optimizer_method(self.unet.parameters(), lr=method_dict["lr"])

    def forward(self, one_image):
        x = torch.unsqueeze(one_image, dim=0)
        latents = self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor
        noise = torch.randn_like(latents, requires_grad=False)
        for t in self.scheduler.timesteps[:-1]:
            noisy_latents = self.scheduler.add_noise(latents, noise, t)

            model_output = self.unet(noise, t)
            # 使用scheduler更新噪声
            noise = self.scheduler.step(model_output, t, noise).prev_sample
        return noise

    def forward(self, noisy_latents, timestep):
        return self.unet(noisy_latents, timestep, self.null_prompt_embeds.to(self.device)).sample

    def training_step(self, batch):
        # print(f"输入图像的shape为: {batch.shape}")  # 检查一下输入的图像的 shape
        total_loss = 0
        for one_data in batch:
            with torch.no_grad():
                x = torch.unsqueeze(one_data, dim=0)
                print(f"提取一张图片输入{x.shape} (升维后)")
                latents = self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor
                # print(f"提取到隐空间的 shape 为 {latents.shape}")
                noise = torch.randn_like(latents)
                # 这里针对每一张照片只生成一个随机的噪声
                timestep = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device)
                # print(f"生成的时刻为 {timestep.shape}")
                noisy_latents = self.scheduler.add_noise(latents, noise, timestep)
                # print(f"加噪之后的隐空间的 shape 为 {noisy_latents.shape}")
            pred_noise = self.forward(noisy_latents.detach(), timestep)
            # print(f"预测噪声的 shape 为 {pred_noise.shape}")
            opt = self.optimizers()
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(pred_noise, noise)
            with torch.autograd.detect_anomaly():
                self.manual_backward(loss)
            opt.step()
        # return total_loss.mean()
        # return total_loss / len(batch)  # 返回该批次平均损失（假设要平均处理，也可以有其他处理方式）

    def configure_optimizers(self):
        return self.optimizer
