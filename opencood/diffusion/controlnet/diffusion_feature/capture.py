import random

import torch
from pytorch_lightning import seed_everything

from opencood.diffusion.controlnet.cldm.model import create_model, load_state_dict


class Capture:
    def __init__(
        self,
        load_model=True,
        basic_dir="controlnet/checkpoints",
        yaml="control_v11f1p_sd15_depth.yaml",
        stable_diffusion_ckpt="v1-5-pruned.ckpt",
        control_net_ckpt="control_v11f1p_sd15_depth_ft.pth",
        seed=-1,
        device=torch.device("cpu"),
    ):
        # basic
        self.device = device
        # diffusion related
        self.basic_dir = basic_dir
        self.yaml = yaml
        self.stable_diffusion_ckpt = stable_diffusion_ckpt
        self.control_net_ckpt = control_net_ckpt
        self.seed = seed

        # else:
        self.strength = 1.0
        self.eta = 1.0
        self.uncond_scale = 5.0
        self.steps = 20
        self.seed = seed
        self.guess_mode = False
        self.only_mid_control = False
        self.a_prompt = "best quality"
        self.n_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"

        if load_model:
            self.load_model()
            self.prepare()

    def load_model(self):
        yaml_path = f"{self.basic_dir}/{self.yaml}"
        stable_diffusion_path = f"{self.basic_dir}/{self.stable_diffusion_ckpt}"
        control_net_path = f"{self.basic_dir}/{self.control_net_ckpt}"
        # cldm.cldm.ControlLDM
        self.model = create_model(yaml_path).cpu()
        self.model.load_state_dict(load_state_dict(stable_diffusion_path, location=self.device), strict=False)
        self.model.load_state_dict(load_state_dict(control_net_path, location=self.device), strict=False)

    def prepare(self):
        if self.seed == -1:
            self.seed = random.randint(0, 65535)
        seed_everything(self.seed)

        self.model.control_model.eval()
        self.model.first_stage_model.eval()
        self.model.cond_stage_model.eval()

        # to cuda
        self.model.to(self.device)
        self.model.control_model.to(self.device)
        self.model.first_stage_model.to(self.device)
        self.model.cond_stage_model.to(self.device)
        self.control_scales = (
            [self.strength * (0.825 ** float(12 - i)) for i in range(13)] if self.guess_mode else ([self.strength] * 13)
        )
        self.model.control_scales = self.control_scales
