"""
Unet Model
Input Image → Conv2D
               ↓
           DownBlock1 → [ResBlock + AdaGN + TimeEmb] → Downsample 'Here to EMB'
               ↓
           DownBlock2 → [ResBlock + AdaGN + TimeEmb] → Downsample
               ↓
           MiddleBlock → [ResBlock + CrossAttention + TimeEmb] 'Here to EMB'
               ↓
           UpBlock1 → [ResBlock + AdaGN + TimeEmb] → Upsample   'Here to EMB'
               ↓
           UpBlock2 → [ResBlock + AdaGN + TimeEmb] → Upsample
               ↓
Output → Conv2D
"""
import math
from diffusers.models.downsampling import downsample_2d
from sympy.solvers.diophantine.diophantine import Linear
from torch import nn
import torch
from torch.nn.functional import upsample
from torchaudio.models.wavernn import ResBlock

"""
timestep need to 'Embedding'
"""
# 正弦编码
def timesetp_embeding(t, dim):
    half_dim = dim // 2
    # 控制嵌入最小频率，默认1w
    max_period = 10000
    emb = math.log(max_period) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = t.float() * emb.unsqueeze(0) # 维度补0
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb

class Unet(nn.Module):
    def __init__(self, T=2000):
        super().__init__()
        self.T = T
        # MLP编码
        self.time_emb = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        self.down_sample = nn.Sequential(
            ResBlock(64, 256), # channel, embeding
            ResBlock(128, 256)
        )
        self.up_sample = nn.Sequential(
            ResBlock(128, 256),
            ResBlock(64, 256)
        )
        self.mid_block = nn.Sequential(
            ResBlock(256, 256)
        )
    def forward(self, x, t):
        time_emb = self.time_emb(t / self.T)
        # 下采样
        for block in self.down_sample:
            x = block(x, time_emb)
            x = downsample_2d(x)
        # 中间层
        x = self.mid_block(x, time_emb)
        # 上采样
        for block in self.up_sample:
            x = upsample(x)
            x = block(x, time_emb)
        return x
