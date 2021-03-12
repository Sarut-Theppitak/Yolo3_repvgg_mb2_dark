import torch
import torch.nn as nn
import numpy as np
from tai_modules import StaticQuantizeConv2d
from .conv_module import Convolutional, Residual_block

# note, underdevelopment ...
class QuantResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6, kernel_size=3, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        mid_channels = in_channels * expand_ratio

        self.conv = nn.Sequential(
            StaticQuantizeConv2d(in_channels, mid_channels, kernel_size=1, padding=0, wBits=8, aBits=8),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            StaticQuantizeConv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=mid_channels, wBits=8, aBits=8),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            StaticQuantizeConv2d(mid_channels, out_channels, kernel_size=1, padding=0, wBits=8, aBits=8),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, input): #: Tensor):# -> Tensor:
        if self.in_channels == self.out_channels and self.stride == 1:
            return input + self.conv(input)
        return self.conv(input)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6, kernel_size=3, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        mid_channels = in_channels * expand_ratio

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, input): #: Tensor):# -> Tensor:
        if self.in_channels == self.out_channels and self.stride == 1:
            return input + self.conv(input)
        return self.conv(input)

#######################################################################################################################################################

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

def quant_conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', StaticQuantizeConv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False, wBits=8, aBits=8))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

            # self.rbr_reparam = StaticQuantizeConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            #                           padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode, wBits=8, aBits=8)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None

            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

            # self.rbr_dense = quant_conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            # self.rbr_1x1 = quant_conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

            print('RepVGG Block, identity = ', self.rbr_identity)


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device) #identity 3x3 kernel
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        # return kernel.detach().cpu().numpy(), bias.detach().cpu().numpy(),
        return kernel.detach(), bias.detach()

optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}
g8_map = {l: 8 for l in optional_groupwise_layers}
g16_map = {l: 16 for l in optional_groupwise_layers}
g32_map = {l: 32 for l in optional_groupwise_layers}

class RepVGG(nn.Module):

    def __init__(self, num_blocks, width_multiplier=None, override_groups_map=None, deploy=False):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        out1 = self.stage2(x)
        out2 = self.stage3(out1)
        out3 = self.stage4(out2)

        return out1, out2, out3


class MobilnetV2(nn.Module): # Customized MobilnetV2

    def __init__(self):
        super(MobilnetV2, self).__init__()

        expand_ratio = 8

        '''
        self.conv1  = StaticQuantizeConv2d(3, 16, kernel_size=3, padding=1, stride=2, wBits=8, aBits=8) # 224->112
        self.conv2  = QuantResBlock(16, 24, stride=2, expand_ratio=expand_ratio) # 112->56
        self.conv3  = QuantResBlock(24, 24, stride=1, expand_ratio=expand_ratio)
        self.conv4  = QuantResBlock(24, 32, stride=2, expand_ratio=expand_ratio) # 56->28
        self.conv5  = QuantResBlock(32, 32, stride=1, expand_ratio=expand_ratio)
        self.conv6  = QuantResBlock(32, 32, stride=1, expand_ratio=expand_ratio)

        self.conv7  = QuantResBlock(32, 64, stride=2, expand_ratio=expand_ratio) # 28->14
        self.conv8  = QuantResBlock(64, 64, stride=1, expand_ratio=expand_ratio)
        self.conv9  = QuantResBlock(64, 64, stride=1, expand_ratio=expand_ratio)
        self.conv10 = QuantResBlock(64, 64, stride=1, expand_ratio=expand_ratio)
        self.conv11 = QuantResBlock(64, 96, stride=1, expand_ratio=expand_ratio) 
        self.conv12 = QuantResBlock(96, 96, stride=1, expand_ratio=expand_ratio)
        self.conv13 = QuantResBlock(96, 96, stride=1, expand_ratio=expand_ratio)

        self.conv14 = QuantResBlock(96,160, stride=2, expand_ratio=expand_ratio) #14->7
        self.conv15 = QuantResBlock(160,160, stride=1, expand_ratio=expand_ratio)
        self.conv16 = QuantResBlock(160,160, stride=1, expand_ratio=expand_ratio)
        self.conv17 = QuantResBlock(160,320, stride=1, expand_ratio=expand_ratio)
        '''


        self.conv1  = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2) # 448->224, 224->112
        self.conv2  = ResBlock(16, 24, stride=2, expand_ratio=expand_ratio) # 224->112, 112->56
        self.conv3  = ResBlock(24, 24, stride=1, expand_ratio=expand_ratio)

        self.conv4  = ResBlock(24, 32, stride=2, expand_ratio=expand_ratio) # 112->56, 56->28
        self.conv5  = ResBlock(32, 32, stride=1, expand_ratio=expand_ratio)
        self.conv6  = ResBlock(32, 32, stride=1, expand_ratio=expand_ratio)

        self.conv7  = ResBlock(32, 64, stride=2, expand_ratio=expand_ratio) # 56->28, 28->14
        self.conv8  = ResBlock(64, 64, stride=1, expand_ratio=expand_ratio)
        self.conv9  = ResBlock(64, 64, stride=1, expand_ratio=expand_ratio)
        self.conv10 = ResBlock(64, 64, stride=1, expand_ratio=expand_ratio)
        self.conv11 = ResBlock(64, 96, stride=1, expand_ratio=expand_ratio) 
        self.conv12 = ResBlock(96, 96, stride=1, expand_ratio=expand_ratio)
        self.conv13 = ResBlock(96, 96, stride=1, expand_ratio=expand_ratio)

        self.conv14 = ResBlock(96,160, stride=2, expand_ratio=expand_ratio) #28->14, 14->7
        self.conv15 = ResBlock(160,160, stride=1, expand_ratio=expand_ratio)
        self.conv16 = ResBlock(160,160, stride=1, expand_ratio=expand_ratio)
        self.conv17 = ResBlock(160,320, stride=1, expand_ratio=expand_ratio)


    def forward(self, x):
        t = x

        t = self.conv1(t)
        t = self.conv2(t)
        t = self.conv3(t)
        t = self.conv4(t)
        t = self.conv5(t)
        t = self.conv6(t)
        t1 = t # (28x28x 32ch)

        t = self.conv7(t)
        t = self.conv8(t)
        t = self.conv9(t)
        t = self.conv10(t)
        t = self.conv11(t)
        t = self.conv12(t)
        t = self.conv13(t)
        t2 = t # (14x14x 96ch)

        t = self.conv14(t)
        t = self.conv15(t)
        t = self.conv16(t)
        t = self.conv17(t)
        t3 = t # (7x7x 320 ch)

        return t1, t2, t3


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()

        # Original Darknet53
        self.__conv = Convolutional(filters_in=3, filters_out=32, kernel_size=3, stride=1, pad=1, norm='bn',
                                    activate='leaky')

        self.__conv_5_0 = Convolutional(filters_in=32, filters_out=64, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        self.__rb_5_0 = Residual_block(filters_in=64, filters_out=64, filters_medium=32)

        self.__conv_5_1 = Convolutional(filters_in=64, filters_out=128, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        self.__rb_5_1_0 = Residual_block(filters_in=128, filters_out=128, filters_medium=64)
        self.__rb_5_1_1 = Residual_block(filters_in=128, filters_out=128, filters_medium=64)

        self.__conv_5_2 = Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        self.__rb_5_2_0 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_1 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_2 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_3 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_4 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_5 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_6 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_7 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)

        self.__conv_5_3 = Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        self.__rb_5_3_0 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_1 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_2 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_3 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_4 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_5 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_6 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_7 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)

        self.__conv_5_4 = Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        self.__rb_5_4_0 = Residual_block(filters_in=1024, filters_out=1024, filters_medium=512)
        self.__rb_5_4_1 = Residual_block(filters_in=1024, filters_out=1024, filters_medium=512)
        self.__rb_5_4_2 = Residual_block(filters_in=1024, filters_out=1024, filters_medium=512)
        self.__rb_5_4_3 = Residual_block(filters_in=1024, filters_out=1024, filters_medium=512)


    def forward(self, x):
        x = self.__conv(x)

        x0_0 = self.__conv_5_0(x) # 416 -> 208
        x0_1 = self.__rb_5_0(x0_0)

        x1_0 = self.__conv_5_1(x0_1) # 208 -> 104
        x1_1 = self.__rb_5_1_0(x1_0)
        x1_2 = self.__rb_5_1_1(x1_1)

        x2_0 = self.__conv_5_2(x1_2) # 104 -> 52
        x2_1 = self.__rb_5_2_0(x2_0)
        x2_2 = self.__rb_5_2_1(x2_1)
        x2_3 = self.__rb_5_2_2(x2_2)
        x2_4 = self.__rb_5_2_3(x2_3)
        x2_5 = self.__rb_5_2_4(x2_4)
        x2_6 = self.__rb_5_2_5(x2_5)
        x2_7 = self.__rb_5_2_6(x2_6)
        x2_8 = self.__rb_5_2_7(x2_7)  # small (52x52, 256ch)

        x3_0 = self.__conv_5_3(x2_8) # 52 -> 26
        x3_1 = self.__rb_5_3_0(x3_0)
        x3_2 = self.__rb_5_3_1(x3_1)
        x3_3 = self.__rb_5_3_2(x3_2)
        x3_4 = self.__rb_5_3_3(x3_3)
        x3_5 = self.__rb_5_3_4(x3_4)
        x3_6 = self.__rb_5_3_5(x3_5)
        x3_7 = self.__rb_5_3_6(x3_6)
        x3_8 = self.__rb_5_3_7(x3_7)  # medium (26x26, 512ch)

        x4_0 = self.__conv_5_4(x3_8) # 26 -> 13
        x4_1 = self.__rb_5_4_0(x4_0)
        x4_2 = self.__rb_5_4_1(x4_1)
        x4_3 = self.__rb_5_4_2(x4_2)
        x4_4 = self.__rb_5_4_3(x4_3)  # large (13x13, 1024ch)

        return x2_8, x3_8, x4_4