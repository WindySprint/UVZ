import torch.nn as nn
import networks.blocks as bs

##### Depth estimation network #####
##### input:256*256*3|output:256*256*1, 2*256*256*64 #####
class d_net1(nn.Module):
    def __init__(self, in_c=3, out_c=1, bc=64):
        super(d_net1, self).__init__()
        self.conv_ini = nn.Sequential(nn.Conv2d(in_c, bc, 3, 1, 1),
                                      nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(bc, bc, 3, 1, 1))

        self.down_conv1 = nn.Sequential(nn.Conv2d(bc, bc, 3, 2, 1),
                                        nn.LeakyReLU(0.1, True))

        self.down_conv2 = nn.Sequential(nn.Conv2d(bc, bc, 3, 2, 1),
                                        nn.LeakyReLU(0.1, True))

        ##############--Dual Attention Module---################
        self.dam1 = bs.DAM(bc,16,256)
        self.dam2 = bs.DAM(bc,16,128)
        self.dam3 = bs.DAM(bc,16,64)

        self.up_conv1 = nn.Sequential(nn.ConvTranspose2d(bc, bc, 4,2,1),
                                      nn.LeakyReLU(0.1, True))

        self.up_conv2 = nn.Sequential(nn.ConvTranspose2d(bc, bc, 4,2,1),
                                      nn.LeakyReLU(0.1, True))

        self.conv_last = nn.Sequential(nn.Conv2d(bc, in_c, 3, 1, 1),
                                       nn.LeakyReLU(0.1, True),
                                       nn.Conv2d(in_c, out_c, 3, 1, 1))

    def forward(self, x):
        e1 = self.conv_ini(x)
        e2 = self.down_conv1(e1)
        e3 = self.down_conv2(e2)
        ####################################
        e3 = e3 + self.dam3(e3)
        ####################################
        d3 = self.up_conv1(e3) + self.dam2(e2)
        d2 = self.up_conv2(d3) + self.dam1(e1)
        out = self.conv_last(d2)

        return out, d2, d3

##### Auxiliary supervision network #####
##### input:256*256*1, 2*256*256*64|output:256*256*3 #####
class d_net2(nn.Module):
    def __init__(self, in_c=1, out_c=3, bc=64):
        super(d_net2, self).__init__()
        self.conv_ini = nn.Sequential(nn.Conv2d(in_c, bc, 3, 1, 1),
                                      nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(bc, bc, 3, 1, 1))

        self.down_conv1 = nn.Sequential(nn.Conv2d(bc, bc, 3, 2, 1),
                                        nn.LeakyReLU(0.1, True))

        self.down_conv2 = nn.Sequential(nn.Conv2d(bc, bc, 3, 2, 1),
                                        nn.LeakyReLU(0.1, True))

        ##############--Residual Skip Block---################
        self.rsb1 = bs.RSB()
        self.rsb2 = bs.RSB()

        self.up_conv1 = nn.Sequential(nn.ConvTranspose2d(bc, bc, 4, 2, 1),
                                      nn.LeakyReLU(0.1, True))

        self.up_conv2 = nn.Sequential(nn.ConvTranspose2d(bc, bc, 4, 2, 1),
                                      nn.LeakyReLU(0.1, True))

        self.conv_last = nn.Sequential(nn.Conv2d(bc, bc, 3, 1, 1),
                                       nn.LeakyReLU(0.1, True),
                                       nn.Conv2d(bc, out_c, 3, 1, 1))

    def forward(self, x, skip1, skip2):
        e1 = self.conv_ini(x)
        r1 = self.rsb1(e1, skip1)
        e2 = self.down_conv1(r1)
        r2 = self.rsb2(e2, skip2)
        e3 = self.down_conv2(r2)
        ####################################
        d3 = self.up_conv1(e3) + e2
        d2 = self.up_conv2(d3) + e1
        out = self.conv_last(d2)
        return out

##### Depth estimation dual-network #####
##### input:256*256*3|output:256*256*1
class d_net(nn.Module):
    def __init__(self):
        super(d_net, self).__init__()
        self.d_net1 = d_net1()
        self.d_net2 = d_net2()

    def forward(self, x, is_train=True):
        if is_train:
            out1, skip1, skip2 = self.d_net1(x)
            out2 = self.d_net2(out1, skip1, skip2)
            return out1, out2
        out, _, _ = self.d_net1(x)
        return out