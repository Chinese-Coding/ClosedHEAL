import pytorch_lightning as L
import torch
from diffusers import StableDiffusionPipeline


class OPV2VDiffusionModel(L.LightningModule):
    def __init__(self, dataloader, optimizer_hype: dict, model_id="stabilityai/stable-diffusion-2-1"):
        super().__init__()
        pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.unet = pipeline.unet
        self.text_encoder = pipeline.text_encoder
        self.scheduler = pipeline.scheduler

        self.dataloder = dataloader
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

    def forward(self, noisy_images, timesteps, text_embeddings):
        return self.unet(noisy_images, timesteps, text_embeddings)

    def training_step(self, batch):
        imgs = batch["ego"]["inputs_m2"]["imgs"]
        print(imgs.shape)
        imgs.to(self.device)
        noise = torch.randn_like(imgs)
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (imgs.shape[0],), device=self.device)
        noisy_images = self.scheduler.add_noise(imgs, noise, timesteps)

        noise_pred = self.forward(noisy_images, timesteps)
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        return loss

    def train_dataloader(self):
        return self.dataloder

    def configure_optimizers(self):
        return self.optimizer
