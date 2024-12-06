import pytorch_lightning as L
import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler


class OPV2VDiffusionModel(L.LightningModule):
    def __init__(self, optimizer_hype: dict, model_id="stabilityai/stable-diffusion-2-1"):
        super().__init__()
        # UNet 去噪网络
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", ignore_mismatched_sizes=True)
        self.unet.train()
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", use_safetensors=True)
        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

        self.optimizer = self._BuildOptimizer(optimizer_hype)

    def _BuildOptimizer(self, hypes):
        method_dict = hypes["optimizer"]
        optimizer_method = getattr(torch.optim, method_dict["core_method"], None)
        if not optimizer_method:
            raise ValueError("{} is not supported".format(method_dict["name"]))
        if "args" in method_dict:
            return optimizer_method(self.unet.parameters(), lr=method_dict["lr"], **method_dict["args"])
        else:
            return optimizer_method(self.unet.parameters(), lr=method_dict["lr"])

    def forward(self, noisy_latents, timestep):
        return self.unet(noisy_latents, timestep)

    def training_step(self, batch):
        print(f"输入图像的shape为: {batch.shape}")  # 检查一下输入的图像的 shape
        latents = self.vae.encode(batch).latent_dist.sample() * self.vae.config.scaling_factor
        print(f"提取到隐空间的 shape 为 {latents.shape}")
        noise = torch.randn_like(latents)
        timestep = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device)
        print(f"生成的时刻为 {timestep.shape}")
        noisy_latents = self.scheduler.add_noise(latents, noise, timestep)
        print(f"加噪之后的隐空间的 shape 为 {noisy_latents.shape}")
        pred_noise = self.forward(noisy_latents, timestep)
        loss = torch.nn.functional.mse_loss(pred_noise, noise)
        return loss

    def configure_optimizers(self):
        return self.optimizer
