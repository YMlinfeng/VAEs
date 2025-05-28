from .modeling_causalvae import CausalVAEModel
from .modeling_wfvae import WFVAEModel
from .modeling_cog3dvae import AutoencoderKLCogVideoX
from .modeling_hunyuan import AutoencoderKLHunyuanVideo
from .modeling_wan import AutoencoderKLWan
from einops import rearrange
from torch import nn