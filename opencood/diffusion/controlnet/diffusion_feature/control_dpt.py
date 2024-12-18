from copy import deepcopy

import einops
import numpy as np
import torch
from PIL import Image

from ..cldm.ddim_hacked import DDIMSampler
from ..utils.util import resize_image, HWC3


class dpt_processor:
    def __init__(self, capturer) -> None:
        # basic
        self.capturer = capturer
        self.ddim_sampler = DDIMSampler(self.capturer.model)

    def depth_normalize(self, depth):
        # following controlnet  1-depth
        depth = depth.astype(np.float64)
        vmin = np.percentile(depth, 2)
        vmax = np.percentile(depth, 85)
        depth -= vmin
        depth /= vmax - vmin
        depth = 1.0 - depth
        depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)
        return depth_image

    def load_dpt(self, dpt_fn):
        if dpt_fn[-3:] in ["npy"]:
            dpt_backup = np.load(dpt_fn)
        else:
            dpt_backup = np.array(Image.open(dpt_fn))
        dpt = deepcopy(dpt_backup)
        # depth normalization -> 0-255 uint8
        dpt = self.depth_normalize(dpt)
        # dpt as network input
        dpt = HWC3(dpt)
        # dpt = cv2.resize(dpt, self.capturer.img_resolution, interpolation=cv2.INTER_LINEAR)
        dpt = resize_image(dpt, self.capturer.img_resolution)
        dpt = np.array(dpt)
        self.H, self.W = dpt.shape[0:2]
        # for visualization
        dpt_backup = dpt_backup[:, :, None].repeat(3, axis=-1).astype(np.float32)
        return dpt_backup, dpt

    def process_given_dpt(self, dpt_backup):
        dpt = deepcopy(dpt_backup)
        # depth normalization -> 0-255 uint8
        dpt = self.depth_normalize(dpt)
        # dpt as network input
        dpt = HWC3(dpt)
        # dpt = cv2.resize(dpt, self.capturer.img_resolution, interpolation=cv2.INTER_LINEAR)
        dpt = resize_image(dpt, self.capturer.img_resolution)
        dpt = np.array(dpt)
        self.H, self.W = dpt.shape[0:2]
        # for visualization
        dpt_backup = dpt_backup[:, :, None].repeat(3, axis=-1).astype(np.float32)
        return dpt_backup, dpt

    def control_input(self, dpt):
        dpt = torch.from_numpy(dpt.copy()).float() / 255.0
        dpt = torch.stack([dpt for _ in range(1)], dim=0)  # 有意思, 为了增加一个维度直接用一个 None 不就行了吗
        dpt = einops.rearrange(dpt, "b h w c -> b c h w").clone()
        dpt = dpt.to(self.capturer.device)
        return dpt

    def control_process(self, dpt, prompt="", final_output=False):
        # noise input -> sd decoding
        cond = {
            "c_concat": [dpt],
            "c_crossattn": [self.capturer.model.get_learned_conditioning([prompt + ", " + self.capturer.a_prompt])],
        }
        un_cond = {
            "c_concat": None if self.capturer.guess_mode else [dpt],
            "c_crossattn": [self.capturer.model.get_learned_conditioning([self.capturer.n_prompt])],
        }
        cond_txt = torch.cat(cond["c_crossattn"], 1).to(self.capturer.device)
        shape = (4, self.H // 8, self.W // 8)
        # conduct diffusion to step 100
        _, intermediates = self.ddim_sampler.sample(
            self.capturer.steps,  # how many diffusion steps
            1,  # generate how many results, we need 1 only
            shape,  # to explain
            cond,  # depth, prompts
            verbose=False,
            eta=self.capturer.eta,
            unconditional_guidance_scale=self.capturer.uncond_scale,
            unconditional_conditioning=un_cond,
            log_every_t=1,
        )  # with depth guidance and generate unconvincing results  -- should not be
        steps = intermediates["step"]
        intermediates = intermediates["x_inter"]
        # the t-th iteration TODO: 论文中说迭代 t 次, 实际上具体实现时则把全部的都计算出来, 然后取 t
        index = round((1000 - self.capturer.t) / (1000 / self.capturer.steps))
        render = intermediates[index]  # 使成为
        # TODO: step: 151, 论文中说取 150 时刻的预测噪声输出, 这里 151, 就ushi把 151 加的噪声输入到 U-Net 进行去噪, 然后就得到在噪声 150 时刻的特征了
        step = steps[index]
        x_samples = self.capturer.model.decode_first_stage(intermediates[-1])

        control_model = self.capturer.model.control_model
        diffusion_model = self.capturer.model.model.diffusion_model
        control = control_model(x=render, hint=torch.cat(cond["c_concat"], 1), timesteps=self.capturer.tlist, context=cond_txt)
        control = [c * scale for c, scale in zip(control, self.capturer.control_scales)]
        with torch.no_grad():
            _, inter_feats = diffusion_model(
                x=render,  # 这里应该就是输入的噪声
                timesteps=self.capturer.tlist,
                context=cond_txt,
                control=control,
                only_mid_control=self.capturer.only_mid_control,
                per_layers=True,
            )
        # 1*c*h*w -> c*h*w
        inter_feats = [i[0].detach() for i in inter_feats]
        if final_output:
            return inter_feats, x_samples
        return inter_feats

    def control_single_dpt(self, dpt_fn="", prompt="", dpt=None):
        # load dpt
        if dpt is None:
            depth, dpt = self.load_dpt(dpt_fn)
        else:
            depth, dpt = self.process_given_dpt(dpt)
        # control input
        dpt = self.control_input(dpt)
        # process
        chw_feat_list = self.control_process(dpt, prompt=prompt)
        return depth, chw_feat_list
