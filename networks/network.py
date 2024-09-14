import functools
import torch.nn as nn
import networks.blocks as bs
from networks.dpm import DPM

##### Depth-guided enhancement network  #####
##### input:256*256*3, 256*256*1|output:256*256*3
class net(nn.Module):
    def __init__(self, in_c=3, out_c=3, bc=64):
        super(net, self).__init__()
        self.conv_ini = nn.Sequential(nn.Conv2d(in_c, bc, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(bc, bc, 3, 1, 1, bias=True))
        self.rb1 = bs.RB()

        self.down_conv1 = nn.Sequential(nn.Conv2d(bc, bc, 3, 2, 1),
                                        nn.LeakyReLU(0.1, True))
        self.down_conv2 = nn.Sequential(nn.Conv2d(bc, bc, 3, 2, 1),
                                        nn.LeakyReLU(0.1, True))

        basic_block = functools.partial(bs.RB, bc)
        self.mrb1 = bs.make_layer(basic_block, 2)
        self.mrb2 = bs.make_layer(basic_block, 8)
        self.mrb3 = bs.make_layer(basic_block, 2)

        # #############--Depth-guided Perception Module---################
        self.dpm1 = DPM(bc)
        self.dpm2 = DPM(bc)
        self.dpm3 = DPM(bc)

        self.up_conv1 = nn.Sequential(nn.Conv2d(bc, 4 * bc, 3, 1, 1), nn.PixelShuffle(2),
                                      nn.LeakyReLU(0.1, True))
        self.up_conv2 = nn.Sequential(nn.Conv2d(bc, 4 * bc, 3, 1, 1), nn.PixelShuffle(2),
                                      nn.LeakyReLU(0.1, True))

        self.rb2 = bs.RB()
        self.conv_last = nn.Sequential(nn.Conv2d(bc, bc, 3, 1, 1, bias=True),
                                       nn.LeakyReLU(0.1, True),
                                       nn.Conv2d(bc, out_c, 3, 1, 1, bias=True))

    def forward(self, x, d=None):
        e1 = self.conv_ini(x)
        e1 = self.rb1(e1)

        e2 = self.down_conv1(e1)
        e2 = self.mrb1(e2)

        e3 = self.down_conv2(e2)
        e3 = self.mrb2(e3)
        e3 = e3 + self.dpm3(e3, d)
        ####################################

        d3 = self.up_conv1(e3) + self.dpm2(e2, d)
        d3 = self.mrb3(d3)

        d2 = self.up_conv2(d3) + self.dpm1(e1, d)
        d2 = self.rb2(d2)

        out = self.conv_last(d2)

        return out