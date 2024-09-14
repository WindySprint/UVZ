# -*- coding: utf-8 -*-out_c
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops.layers.torch import Rearrange
from einops import rearrange
from timm.models.layers import trunc_normal_

import networks.blocks as bs

logger = logging.getLogger(__name__)

def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(1/3)

# Use the avgpool layer to unify the values of each region in the depth map
def avg_map(d, size):
    # Set size to kernel size and
    # stride
    avgpool = nn.AvgPool2d(size, stride=size, padding=0)
    avg_d = avgpool(d)
    # print(avg_d.is_cuda)
    # print(torch.ones((size, size)).is_cuda)
    # Expand the avgpool value to the size*size region
    matrix3 = torch.ones((size, size))
    uni_d = torch.kron(avg_d, matrix3.to(avg_d.device))
    # Map the extended matrix back to the original matrix
    temp_d = d.clone()
    temp_d[:,:,:uni_d.size(2),:uni_d.size(3)] = uni_d
    d = temp_d.clone()
    return d

class W_SW_Attention(nn.Module):
    def __init__(self, in_c, out_c, hc, w_size, type):
        super(W_SW_Attention, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.hc = hc
        self.scale = self.hc ** -0.5
        self.n_heads = in_c // hc
        self.w_size = w_size
        self.type = type
        self.embedding_layer = nn.Linear(self.in_c, 3 * self.in_c, bias=True)
        self.linear = nn.Linear(self.in_c, self.out_c)

        self.relative_position_params = nn.Parameter(
            torch.zeros((2 * w_size - 1) * (2 * w_size - 1), self.n_heads))
        trunc_normal_(self.relative_position_params, std=.02)

        self.relative_position_params = torch.nn.Parameter(
            self.relative_position_params.view(2 * w_size - 1, 2 * w_size - 1,
                                               self.n_heads).transpose(1, 2).transpose(0, 1))

    def generate_mask(self, w, p, shift):
        attn_mask = torch.zeros(w, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, emb):
        if self.type != 'W': emb = torch.roll(emb, shifts=(-(self.w_size // 2), -(self.w_size // 2)), dims=(1, 2))
        emb = rearrange(emb, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.w_size, p2=self.w_size)
        h_windows = emb.size(1)
        w_windows = emb.size(2)
        # sqaure validation
        assert h_windows == w_windows

        emb = rearrange(emb, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.w_size, p2=self.w_size)
        qkv = self.embedding_layer(emb)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.hc).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, self.w_size, shift=self.w_size // 2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.w_size)

        if self.type != 'W': output = torch.roll(output, shifts=(self.w_size // 2, self.w_size // 2),
                                                 dims=(1, 2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.w_size) for j in range(self.w_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.w_size - 1
        # negative is allowed
        return self.relative_position_params[:, relation[:, :, 0].long(), relation[:, :, 1].long()]

#############--Swin Transformer Block--################
class STB(nn.Module):
    def __init__(self, in_c, out_c, hc, w_size, type):
        super(STB, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        assert type in ['W', 'SW']
        self.type = type

        # print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(in_c)
        self.w_sw_attn = W_SW_Attention(in_c, in_c, hc, w_size, self.type)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(in_c)
        self.mlp = nn.Sequential(
            nn.Linear(in_c, 4 * in_c),
            nn.GELU(),
            nn.Linear(4 * in_c, out_c),
        )

    def forward(self, emb):
        emb = emb + self.w_sw_attn(self.ln1(emb))
        emb = emb + self.mlp(self.ln2(emb))
        return emb

#############--Multi-kernel convolutional block--################
class MKCB(nn.Module):
    def __init__(self, bc):
        super(MKCB, self).__init__()
        self.conv_ini = nn.Sequential(nn.Conv2d(bc, bc, 3, 1, 1),
                                      nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(bc, bc, 3, 1, 1))
        basic_block = functools.partial(bs.RB, bc)
        self.mrb = bs.make_layer(basic_block, 4)        
        # Set 1*1, 3*3, 5*5 rate and convolution
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate3 = torch.nn.Parameter(torch.Tensor(1))
        self.rate5 = torch.nn.Parameter(torch.Tensor(1))
        init_rate_half(self.rate1)
        init_rate_half(self.rate3)
        init_rate_half(self.rate5)                
        self.conv1 = nn.Sequential(nn.Conv2d(bc, bc, 1, 1, 0),
                                        nn.LeakyReLU(0.1, True))
        self.conv3 = nn.Sequential(nn.Conv2d(bc, bc, 3, 1, 1),
                                        nn.LeakyReLU(0.1, True))
        self.conv5 = nn.Sequential(nn.Conv2d(bc, bc, 5, 1, 2),
                                        nn.LeakyReLU(0.1, True))
        self.out = nn.Sequential(nn.Conv2d(3*bc, bc, kernel_size=1),
                                 nn.LeakyReLU(0.1, True),
                                 nn.Conv2d(bc, bc, 3, 1, 1))
        
    def forward(self, x, d1 = None, d3 = None, d5 = None):
        x = self.conv_ini(x)
        r = self.mrb(x)
        # Multiply the convolution result with the corresponding depth map
        x1 = self.conv1(r)*d1
        x3 = self.conv3(r)*d3
        x5 = self.conv5(r)*d5
        out = self.out(torch.cat([self.rate1 * x1, self.rate3 * x3, 
                                   self.rate5 * x5], 1))                
        return out

#############--Depth Perception Module--################
class DPM(nn.Module):
    def __init__(self, bc):
        super(DPM, self).__init__()
        self.b_channels = bc
        self.ini = nn.Sequential(Rearrange('b c h w -> b h w c'), nn.LayerNorm(bc))
        self.stb1 = STB(bc, bc, 16, 8, 'W')
        self.stb2 = STB(bc, bc, 16, 8, 'SW')
        self.end = nn.Sequential(nn.LayerNorm(bc), Rearrange('b h w c -> b c h w'))
        
        self.mkcb = MKCB(bc)

    def forward(self, x, d=None):
        # Adapt to corresponding scale
        d = F.interpolate(d, size=[x.size(2), x.size(3)], mode='nearest')
        d_rev = 1.0 - d
        # Swin transformer
        st = self.ini(x)
        st = self.stb1(st)
        st = self.stb2(st)
        st = self.end(st)*d_rev
        
        # Multi-kernel convolutional block
        # Obtain different avgpool depth maps
        d3 = avg_map(d, 3)
        d5 = avg_map(d, 5)
        mk = self.mkcb(x, d, d3, d5)
        out = st + mk
        x = x + out
        return x

