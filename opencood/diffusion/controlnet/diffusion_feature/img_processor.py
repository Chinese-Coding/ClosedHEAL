import torch
from torch import nn


class ImageProcessor(nn.Module):
    def __init__(self, capturer) -> None:
        super().__init__()
        self.capturer = capturer

    def forward(self, image):
        latent = self.capturer.model.encode_first_stage(image)
        latent = self.capturer.model.get_first_stage_encoding(latent).detach()
        # TODO: 感觉这里可能有问题
        # fmt: off
        cond = {"c_crossattn": [
            self.capturer.model.get_learned_conditioning(["" + ", " + self.capturer.a_prompt]),
            self.capturer.model.get_learned_conditioning(["" + ", " + self.capturer.a_prompt]),
            self.capturer.model.get_learned_conditioning(["" + ", " + self.capturer.a_prompt]),
            self.capturer.model.get_learned_conditioning(["" + ", " + self.capturer.a_prompt]),
        ]}
        cond_txt = torch.cat(cond["c_crossattn"], 0).to(self.capturer.device)
        # fmt: on
        tlist = self.capturer.get_tlist()
        noise = torch.randn_like(latent)
        noise_latent = self.capturer.model.q_sample(latent, tlist, noise)
        pred_noise, inter_features = self.capturer.model.model.diffusion_model(
            x=noise_latent,
            timesteps=tlist,
            context=cond_txt,
            control=None,
            only_mid_control=self.capturer.only_mid_control,
            per_layers=True,
        )
        return noise, pred_noise
