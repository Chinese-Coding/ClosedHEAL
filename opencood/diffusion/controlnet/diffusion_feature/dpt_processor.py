from copy import deepcopy

import numpy as np
import torch
from torch import nn

from opencood.diffusion.controlnet.cldm.ddim_hacked import DDIMSampler
from opencood.diffusion.controlnet.utils.util import resize_image, HWC3


def depth_normalize(depth):
    depth = depth.astype(np.float64)
    # 这两个操作用于计算深度图的一个有效范围，以避免极端值（如异常噪声）对归一化处理的影响，从而将大部分有效的深度信息用于后续处理。
    # 取出深度图中的第 2 和第 85 百分位的值
    vmin, vmax = np.percentile(depth, 2), np.percentile(depth, 85)
    depth -= vmin  # 首先将所有的深度值减去 vmin，使得最小值变为零。
    depth /= vmax - vmin  # 接着将所有的深度值除以 (vmax - vmin)，使得归一化后的深度值范围在 [0, 1] 之间。
    # 这一操作将深度值进行反转，原本较小的深度值变为较大的值，较大的深度值变为较小的值。这样做的目的是为了使得“近距离”在视觉上更加突出（通常在深度图中，深度越小表示越近）。
    depth = 1.0 - depth
    # 将归一化后的深度值（范围 [0, 1]）映射到 8-bit 图像的值域 [0, 255]。
    # clip(0, 255) 确保输出的像素值不会超出这个范围，最后将其转换为 np.uint8 类型，适合用于图像显示。
    depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)
    return depth_image


class DPTProcessor(nn.Module):
    def __init__(self, capturer):
        super().__init__()
        self.capturer = capturer
        self.ddim_sampler = DDIMSampler(self.capturer.model)

    def process_given_dpt(self, dpt_backup):
        dpt = deepcopy(dpt_backup)
        # depth normalization -> 0-255 uint8
        dpt = depth_normalize(dpt)
        # dpt as network input
        # TODO: 这里也许可以不用转换成 HWC 格式的形式, 感觉 HWC 这种形式不是很常见啊
        dpt = HWC3(dpt)  # 这里将dpt转换为 HWC 的形式, 因为下面那个函数需要输入的参数是HWC的
        # dpt = cv2.resize(dpt, self.capturer.img_resolution, interpolation=cv2.INTER_LINEAR)
        dpt = resize_image(dpt, self.capturer.img_resolution)
        dpt = np.array(dpt)
        self.H, self.W = dpt.shape[0:2]
        # for visualization
        dpt_backup = dpt_backup[:, :, None].repeat(3, axis=-1).astype(np.float32)
        return dpt_backup, dpt

    def forward(self, dpt, prompt="", final_output=False):
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
        # TODO: 明天这里需要和学长商量一下, 这里有坑啊.
        #       他是只用推理, 不用训练. 但是我需要训练啊, 这里用推理, 下面我又要训练, 感觉好矛盾啊
        #       如果这样可行的话, 那就只需要把 `round((1000 - self.capturer.t) / (1000 / self.capturer.steps))`
        #       这部分该一改, 不再取固定的值, 也是每次训练时取一个随机的值 是否要和前面的 image 一样)
        # 在 control 的控制条件下, 通过 ddim_samler 逐步从噪声中去噪, 获得每一步的去噪结果
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

        x_inter, noise = intermediates["x_inter"], intermediates["noise"]
        # the t-th iteration TODO: 论文中说迭代 t 次, 实际上具体实现时则把全部的都计算出来, 然后取 t
        tlist = self.capturer.get_tlist()
        t = tlist[0]
        index = round((1000 - t) / (1000 / self.capturer.steps))
        render = x_inter[index]  # 使成为
        add_noise = noise[index]
        # TODO: step: 151, 论文中说取 150 时刻的预测噪声输出, 这里 151, 就ushi把 151 加的噪声输入到 U-Net 进行去噪, 然后就得到在噪声 150 时刻的特征了
        x_samples = self.capturer.model.decode_first_stage(x_inter[-1])

        control_model = self.capturer.model.control_model
        diffusion_model = self.capturer.model.model.diffusion_model
        control = control_model(x=render, hint=torch.cat(cond["c_concat"], 1), timesteps=self.capturer.tlist, context=cond_txt)
        control = [c * scale for c, scale in zip(control, self.capturer.control_scales)]
        pred_noise, inter_feats = diffusion_model(
            x=render,  # 这里应该就是输入的噪声
            timesteps=self.capturer.tlist,
            context=cond_txt,
            control=control,
            only_mid_control=self.capturer.only_mid_control,
            per_layers=True,
        )
        return add_noise, pred_noise
        # # 1*c*h*w -> c*h*w
        # inter_feats = [i[0].detach() for i in inter_feats]
        # if final_output:
        #     return inter_feats, x_samples
        # return inter_feats
