from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import AutoencoderKLCogVideoX
from ..registry import ModelRegistry

# 定义适配器类
@ModelRegistry.register("Cog3DVAE")
class WrappedCog3DVAE(AutoencoderKLCogVideoX):
    def __init__(self, *args, **kwargs): # 透传参数，分别接受位置参数和关键字参数
        super().__init__(*args, **kwargs)

    def get_encoder(self):
        modules = [self.encoder]
        if self.quant_conv is not None:
            modules.append(self.quant_conv)
        return modules

    def get_decoder(self):
        modules = [self.decoder]
        if self.post_quant_conv is not None:
            modules.append(self.post_quant_conv)
        return modules
    
    def get_last_layer(self):
        if hasattr(self.decoder.conv_out, "conv"):
            return self.decoder.conv_out.conv.weight
        else:
            return self.decoder.conv_out.weight
    
    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample
        return dec, posterior, None # 如果将来训练代码需要 wavelet loss

# 手动注册别名
# ModelRegistry.register("Cog3DVAE")(WrappedCog3DVAE)