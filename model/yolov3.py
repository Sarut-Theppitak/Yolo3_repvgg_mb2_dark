import sys
sys.path.append("..")

import os
AbsolutePath = os.path.abspath(__file__)           
SuperiorCatalogue = os.path.dirname(AbsolutePath)   
BaseDir = os.path.dirname(SuperiorCatalogue)       
sys.path.insert(0,BaseDir)                          

import torch.nn as nn
import torch
import torchvision
from model.backbones import MobilnetV2, RepVGG, Darknet53
from model.yolo_fpn import FPN_YOLOV3
from model.yolo_head import Yolo_head
from model.conv_module import Convolutional
import numpy as np
from utils.tools import *
from torchsummary import summary 


class Yolov3(nn.Module):

    optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    g2_map = {l: 2 for l in optional_groupwise_layers}
    g4_map = {l: 4 for l in optional_groupwise_layers}
    g8_map = {l: 8 for l in optional_groupwise_layers}
    g16_map = {l: 16 for l in optional_groupwise_layers}
    g32_map = {l: 32 for l in optional_groupwise_layers}
    A_num_blocks = [2, 4, 14, 1]
    B_num_blocks = [4, 6, 16, 1]
    S_num_blocks = [2, 4, 14, 1]
    w_A0 = [0.75, 0.75, 0.75, 2.5] # [1280, 192, 96]
    w_A1 = [1, 1, 1, 2.5]  # [1280, 256, 128]
    w_S0 = [0.5, 0.5, 0.5, 1] #[512, 128, 64]
    w_S1 = [0.75, 0.75, 0.75, 2]  #  [1024, 192, 96]    
    w_Ss = [0.375, 0.375, 0.375, 0.75] # [384, 96, 48]         

    def __init__(self, cfg, init_weights=True, deploy=False): 
        super(Yolov3, self).__init__()

        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__out_channel = cfg.MODEL["ANCHORS_PER_SCLAE"] * (self.__nC + 5)

        self.__backbone = MobilnetV2()
        #self.__fpn = FPN_YOLOV3(fileters_in=[1024, 512, 256],
        #                        fileters_out=[self.__out_channel, self.__out_channel, self.__out_channel])
        self.__fpn = FPN_YOLOV3(fileters_in=[320, 96, 32],
                                fileters_out=[self.__out_channel, self.__out_channel, self.__out_channel])

        ##################################################################################################
        # self.__backbone = RepVGG(num_blocks=self.S_num_blocks, width_multiplier=self.w_Ss, 
        #                          override_groups_map=None, deploy=deploy)
        # self.__fpn = FPN_YOLOV3(fileters_in=[384, 96, 48],
        #                         fileters_out=[self.__out_channel, self.__out_channel, self.__out_channel])
        ##################################################################################################
        # self.__backbone = Darknet53()
        # self.__fpn = FPN_YOLOV3(fileters_in=[1024, 512, 256],
        #                         fileters_out=[self.__out_channel, self.__out_channel, self.__out_channel])
        ###################################################################################################

        # small
        self.__head_s = Yolo_head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        # medium
        self.__head_m = Yolo_head(nC=self.__nC, anchors=self.__anchors[1], stride=self.__strides[1])
        # large
        self.__head_l = Yolo_head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])

        if init_weights:
            self.__init_weights()


    def forward(self, x, valid=False):
        out = []

        x_s, x_m, x_l = self.__backbone(x)
        x_s, x_m, x_l = self.__fpn(x_l, x_m, x_s)

        out.append(self.__head_s(x_s, valid=valid))
        out.append(self.__head_m(x_m, valid=valid))
        out.append(self.__head_l(x_l, valid=valid))
        p, p_d = list(zip(*out))

        if self.training:
            return p, p_d  # small, medium, large list of [(bs,nG,nG,nA,8),(bs,nG,nG,nA,8), (bs,nG,nG,nA,8)] small(1/8) medium large
        else:
            if valid:
                return p, p_d  
            else:
                #if i want only the last layers comment out below
                # p_d =  [p_d[1], p_d[2]]
                return p, torch.cat(p_d, 0)   # a list of [(all boxes,8), (all boxes,8), (all boxes,8)] >> so torch.cat(p_d, 0) = (total boxes, 8)
        # boxes are in xywh

    def __init_weights(self):

        " Note ：nn.Conv2d nn.BatchNorm2d'initing modes are uniform "
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)

                print("initing {}".format(m))

    # def load_weight_mobilev2(self, path):
    #     state_dict = torch.load(weightfile)
    #     self.load_state_dict(state_dict)


if __name__ == '__main__':
    import config.yolov3_config_yoloformat as cfg
    net = Yolov3(cfg=cfg)
    print(net)
    # weightfile = 'weight/lw_movb2_fp_nopretrain03Feb2021/best.pt'
    # print(net._Yolov3__backbone.conv1.weight[0,0,0])
    # # for name, param in net.state_dict().items():
    # #     print(name)
    # # print(net.state_dict())
    # net.load_weight_mobilev2(weightfile)
    # print(net._Yolov3__backnone.conv1.weight[0,0,0])

    net.eval()
    in_img = torch.randn(12, 3, 384, 384)
    p, p_d = net(in_img)

    if net.training:
        for i in range(3):
            print(p[i].shape)
            print(p_d[i].shape)
    else:
        for i in range(3):
            print(p[i].shape)
        print(p_d.shape)