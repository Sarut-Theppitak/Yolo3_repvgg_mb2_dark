import torch.nn as nn
import torch

class Yolo_head(nn.Module):
    def __init__(self, nC, anchors, stride):
        super(Yolo_head, self).__init__()

        self.__anchors = anchors  # our anchors boxes is the grid scale to this stride. we define it in the config file  ex. yolo3_config_voc.py
        self.__nA = len(anchors)
        self.__nC = nC
        self.__stride = stride


    def forward(self, p, valid=False):
        bs, nG = p.shape[0], p.shape[-1]
        p = p.view(bs, self.__nA, 5 + self.__nC, nG, nG).permute(0, 3, 4, 1, 2)  #(bs,nG,nG,nA,8)

        p_de = self.__decode(p.clone(), valid=valid)

        return (p, p_de)  # p is the raw output from yolo head, p_de is already sigmoid or exp() and also in pixel already. it is xywh in grid scale
        # p_de >> if not training with shape of (all bbx in this grid, 8)
        # p_de >> if training with shape of (bs,nG,nG,nA,8) same with p

    def __decode(self, p, valid=False):
        batch_size, output_size = p.shape[:2]

        device = p.device
        stride = self.__stride
        anchors = (1.0 * self.__anchors).to(device)

        conv_raw_dxdy = p[:, :, :, :, 0:2]
        conv_raw_dwdh = p[:, :, :, :, 2:4]
        conv_raw_conf = p[:, :, :, :, 4:5]
        conv_raw_prob = p[:, :, :, :, 5:]

        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3, 1).float().to(device)

        pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)
        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)
        
        if self.training:
            return pred_bbox
        else:
            if valid:
                return pred_bbox
            else:
                return pred_bbox.view(-1, 5 + self.__nC) 