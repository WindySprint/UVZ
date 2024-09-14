import torch
import torch.nn as nn
import torch.nn.functional as F

##############--- Channel Attention ---################
class CA(nn.Module):
    def __init__(self, bc, r):
        super(CA,self).__init__()
        self.avgp = nn.AdaptiveAvgPool2d(1)
        self.maxp = nn.AdaptiveMaxPool2d(1)
        self.clrc1 = nn.Sequential(
            nn.Conv2d(bc, bc // r, 1, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(bc // r, bc, 1, bias=False)
        )
        self.clrc2 = nn.Sequential(
            nn.Conv2d(bc, bc // r, 1, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(bc // r, bc, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a1 = self.avgp(x)
        a1 = self.clrc1(a1)
        m1 = self.maxp(x)
        m1 = self.clrc2(m1)
        out = self.sigmoid(a1 + m1)
        return out

##############--- Self Attention ---################
class SA(nn.Module):
    def __init__(self, bc, dim):
        super().__init__()
        self.bc = bc
        self.q = nn.Conv1d(bc, bc // 3, 1)
        self.k = nn.Conv1d(bc, bc // 3, 1)
        self.v = nn.Conv1d(bc, bc, 1)

        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.avg_pool = nn.AdaptiveAvgPool2d(dim//2)
        self.deconv = nn.ConvTranspose2d(in_channels=bc, out_channels=bc,kernel_size=4,stride=2,padding=1,output_padding=0,bias=True)
    def forward(self, input):

        input= self.avg_pool(input)
        shape = input.shape
        flatten = input.view(1, self.bc, -1)
        query = self.q(flatten).permute(0, 2, 1)
        key = self.k(flatten)
        value = self.v(flatten)
        query_key = torch.matmul(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.matmul(value, attn)
        attn = attn.view(*shape)
        out = self.gamma * attn + input
        out = self.deconv(out)
        return out

##############--- Dual-attention Module (DAM) ---################
class DAM (nn.Module):
    def __init__(self, bc, r, dim):
        super(DAM, self).__init__()
        self.ca = CA(bc, r)
        self.sa = SA(bc, dim)

    def forward(self, x):
        x1 = x * self.ca(x)
        x1 = x1 * self.sa(x1)
        out = x + x1
        return out

##############--- Multi-blocks ---################
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

##############--- Residual Block (RB) ---################
class RB(nn.Module):
    def __init__(self, bc=64):
        super(RB, self).__init__()
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(bc, bc, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(bc, bc, 3, 1, 1)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(bc, bc, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(bc, bc, 3, 1, 1)
        )
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(bc, bc, 1),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(bc, bc, 1)
        # )

    def forward(self, x):
        x1 = self.conv3_2(self.conv3_1(x))
        x = x + x1
        return x

##############--- Residual Skip Block (RSB) ---################
class RSB(nn.Module):
    def __init__(self, bc=64):
        super(RSB, self).__init__()
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(bc, bc, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(bc, bc, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(bc, bc, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(bc, bc, kernel_size=1, bias=False)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(bc, bc, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(bc, bc, kernel_size=1, bias=False)
        )

    def forward(self, x, skip_x):
        x1 = self.conv1(self.conv3_1(x))
        x1 = x1 + self.conv3_2(skip_x)
        x = x + x1
        return x