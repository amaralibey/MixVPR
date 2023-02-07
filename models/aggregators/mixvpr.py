import torch
import torch.nn.functional as F
import torch.nn as nn

from einops.layers.torch import Rearrange
from einops import rearrange
import numpy as np


class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, p=0.0, mlp_ratio=1):
        super().__init__()
        # self.gamma = nn.Parameter(torch.ones(1024), requires_grad=True)
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            # nn.Dropout(p),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )
        
        # for m in self.modules():
        #     if isinstance(m, (nn.Linear)):
        #         nn.init.trunc_normal_(m.weight, std=0.02)
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        return x + self.mix(x)


class MixVPR(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 in_h=20,
                 in_w=20,
                 out_channels=512,
                 mix_depth=1,
                 dropout=0.0,
                 mlp_ratio=1,
                 row_out=4,
                 ) -> None:
        super().__init__()

        self.in_h=in_h
        self.in_w=in_w
        self.in_channels=in_channels
        self.out_channels=out_channels

        self.mix_depth=mix_depth
        self.dropout=dropout
        self.mlp_ratio=mlp_ratio
        self.row_out=row_out


        hw = in_h*in_w
        self.mix = nn.Sequential(*[
                        FeatureMixerLayer(hw, p=dropout, mlp_ratio=mlp_ratio)
                        for _ in range(self.mix_depth)
                    ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, self.row_out, bias=False)
    def forward(self, x):
        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0,2,1)
        x = self.channel_proj(x) 
        x = x.permute(0,2,1)
        x = self.row_proj(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x


def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')

def main():
    x = torch.randn(1, 2048, 14, 14)
    agg = MixVPR(
                in_channels=2048,
                in_h=14,
                in_w=14,
                out_channels=1024,
                mix_depth=4,
                dropout=0.0,
                mlp_ratio=2,
                row_out=4)

    print_nb_params(agg)

    r = agg(x)
    print(r.shape)
    
if __name__ == '__main__':
    main()
