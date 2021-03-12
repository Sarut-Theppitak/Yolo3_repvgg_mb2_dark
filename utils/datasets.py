# coding=utf-8
import os
import sys

sys.path.append("..")
sys.path.append("../utils")
AbsolutePath = os.path.abspath(__file__)           
SuperiorCatalogue = os.path.dirname(AbsolutePath)   
BaseDir = os.path.dirname(SuperiorCatalogue)       
sys.path.insert(0,BaseDir)    

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import random
import utils.data_augment as dataAug
import utils.tools as tools

class BaseFunction:
    def creat_label(self, bboxes):
        """
        Label assignment. For a single picture all GT box bboxes are assigned anchor.
        1、Select a bbox in order, convert its coordinates("xyxy") to "xywh"; and scale bbox'
           xywh by the strides.
        2、Calculate the iou between the each detection layer'anchors and the bbox in turn, and select the largest
            anchor to predict the bbox.If the ious of all detection layers are smaller than 0.3, select the largest
            of all detection layers' anchors to predict the bbox.

        Note :
        1、The same GT may be assigned to multiple anchors. And the anchors may be on the same or different layer. Pai : even the same cell
        2、The total number of bboxes may be more than it is, because the same GT may be assigned to multiple layers
        of detection.

        """
        anchors = np.array(self.cfg_MODEL["ANCHORS"])
        strides = np.array(self.cfg_MODEL["STRIDES"])
        train_output_size = self.img_size / strides
        anchors_per_scale = self.cfg_MODEL["ANCHORS_PER_SCLAE"]

        label = [np.zeros((int(train_output_size[i]), int(train_output_size[i]), anchors_per_scale, 6+self.num_classes))
                                                                      for i in range(3)]
        # label = [np,np.np] each have size = grid,grid, anchors_per_scale, 6+self.num_classes
        for i in range(3):
            label[i][..., 5] = 1.0

        bboxes_xywh = [np.zeros((150, 4)) for _ in range(3)]   # Darknet the max_num is 30
        bbox_count = np.zeros((3,))

        for bbox in bboxes: # start one by one gt box
            bbox_coor = bbox[:4]
            bbox_class_ind = int(bbox[4])
            bbox_mix = bbox[5]

            # onehot
            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            one_hot[bbox_class_ind] = 1.0
            one_hot_smooth = dataAug.LabelSmooth()(one_hot, self.num_classes)

            # convert "xyxy" to "xywh"
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                                        bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            # print("bbox_xywh: ", bbox_xywh)

            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]   # scale gt box to grid unit sp shape = [3,4] each row is at diferent scale. 1 grind size is 8, 16 ,32 

            iou = []
            exist_positive = False
            for i in range(3):  # we have 3 scales
                anchors_xywh = np.zeros((anchors_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5  # 0.5 for compensation
                anchors_xywh[:, 2:4] = anchors[i]

                iou_scale = tools.iou_xywh_numpy(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    # Bug : When multiple gt bboxes correspond to the same anchor, the anchor is assigned to the last bbox by default
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh  # Pai: this means the 3 anchors in that cell can be responsible to the boxes. Not choose one in this code
                    label[i][yind, xind, iou_mask, 4:5] = 1.0   # You seeeeeeeeeeee the confidence for the gt = 1 kuayyyyyyyyyyyy finally i got the answer
                    label[i][yind, xind, iou_mask, 5:6] = bbox_mix
                    label[i][yind, xind, iou_mask, 6:] = one_hot_smooth

                    bbox_ind = int(bbox_count[i] % 150)  # BUG : 150 is a prior value, memory consumption is large
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / anchors_per_scale)
                best_anchor = int(best_anchor_ind % anchors_per_scale)

                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:6] = bbox_mix
                label[best_detect][yind, xind, best_anchor, 6:] = one_hot_smooth

                bbox_ind = int(bbox_count[best_detect] % 150)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh

        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

class YoloDataset(Dataset, BaseFunction):
    def __init__(self, cfg, data_dir, img_size=416, augmentation=True):
        self.cfg_MODEL = cfg.MODEL
        self.img_size = img_size  # For Multi-training
        self.classes = cfg.DATA["CLASSES"]
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))
        self.augmentation = augmentation
        img_dir = os.path.join(data_dir,"images")
        anno_dir = os.path.join(data_dir,"labels")
        img_names = os.listdir(img_dir)
        self.img_paths = [os.path.join(img_dir,img_name) for img_name in img_names]
        self.annos = []
        self.albu_aug = dataAug.albumentations_augmentation()
        self.random_erasor_I = dataAug.get_random_eraser_I()
        self.random_erasor_O = dataAug.get_random_eraser_O()
        for i, img_name in enumerate(img_names):
            anno_path = os.path.join(anno_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.tif', '.txt'))
            try:
                boxes = np.loadtxt(anno_path).reshape(-1, 5)   # eventiought the file is blank we will get np.array([])
                self.annos.append(boxes)
            except Exception as e:
                self.annos.append(np.array([]))
                print(f'system error massage: {e}')
                print(f"Annotation file is not existed (train dataset): '{anno_path}'.")
        # the boxese is in xyxy format 

    def __len__(self):
        return  len(self.img_paths)

    def __getitem__(self, item):

        img_org, bboxes_org = self.__parse_data(self.img_paths[item], self.annos[item])
        img_org = img_org.transpose(2, 0, 1)  # HWC->CHW
        
        item_mix = random.randint(0, len(self.annos)-1)
        img_mix, bboxes_mix = self.__parse_data(self.img_paths[item_mix], self.annos[item_mix])
        img_mix = img_mix.transpose(2, 0, 1)
        img, bboxes = dataAug.Mixup(p=0.5)(img_org, bboxes_org, img_mix, bboxes_mix)
        del img_org, bboxes_org, img_mix, bboxes_mix

        # Without mixing. or we can set p=1 in the dataAug.Mixup(p=1)
        # img = img_org
        # bboxes = np.concatenate([bboxes_org, np.full((len(bboxes_org), 1), 1.0)], axis=1)
        # del img_org, bboxes_org
        
        # for debuging  we will see this 2 times in 1 loop if we set batch = 2
        # new_boxes = np.copy(bboxes)
        # im = np.copy(img)
        # im = im.transpose(1, 2, 0)
        # im = im[:,:,[2,1,0]]  
        # im = im.astype('float32')
        # im = cv2.UMat(im)
        # for box in new_boxes:
        #     box = box.astype('float32')
        #     cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]) , (0,255,0), 2)
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     color = (255, 0, 0) 
        #     cv2.putText(im , str(box[4]),(int(box[0]-10),int(box[1])), font, .5,color ,2,cv2.LINE_AA)
        # cv2.imshow('test mix', im) 
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.creat_label(bboxes)

        img = torch.from_numpy(img).float()
        label_sbbox = torch.from_numpy(label_sbbox).float()
        label_mbbox = torch.from_numpy(label_mbbox).float()
        label_lbbox = torch.from_numpy(label_lbbox).float()
        sbboxes = torch.from_numpy(sbboxes).float()
        mbboxes = torch.from_numpy(mbboxes).float()
        lbboxes = torch.from_numpy(lbboxes).float()

        return img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __parse_data(self, img_path, anno):

        img = cv2.imread(img_path)  # H*W*C and C=BGR
        img_h, img_w = img.shape[0], img.shape[1]
        assert img is not None, 'File Not Found ' + img_path
        # permutation = [4, 0, 1, 2, 3]
        # idx = np.empty_like(permutation)
        # idx[permutation] = np.arange(len(permutation))
        # anno = anno[:, idx]
        if len(anno) == 0:
            img, bboxes = dataAug.Resize((self.img_size, self.img_size), True)(np.copy(img), np.copy(anno))
            return img, bboxes

        bboxes = anno[:, [1,2,3,4,0]]    # return a copy so it will not change the original anno which we will use inthe next epoch
        bboxes[:,:4:2] = bboxes[:,:4:2] * img_w
        bboxes[:,1:4:2] = bboxes[:,1:4:2] * img_h
        bboxes[:,0:4] = dataAug.xywh2xyxy_np(bboxes[:,0:4])
        bboxes = np.round(bboxes)

        if self.augmentation:
            img, bboxes = dataAug.RandomHorizontalFilp()(np.copy(img), np.copy(bboxes))
            img, bboxes = dataAug.RandomCrop()(np.copy(img), np.copy(bboxes))
            img, bboxes = dataAug.RandomAffine()(np.copy(img), np.copy(bboxes))

            bboxes = np.where(bboxes < 0.0, 0.0, bboxes)
            bboxes[:,2] = np.where(bboxes[:,2] > img_w, img_w, bboxes[:,2])
            bboxes[:,3] = np.where(bboxes[:,3] > img_h, img_h, bboxes[:,3])
            sample = self.albu_aug(image=np.copy(img), bboxes=np.copy(bboxes))
            img, bboxes = sample['image'], sample['bboxes']
            bboxes = np.array(bboxes)
            if len(bboxes) == 0:
                bboxes = np.array(bboxes)
                bboxes = bboxes.reshape(-1, 5) # so that it will not error when go to Resize when it is blank

            # Apply random erasor
            # if len(bboxes) != 0:
            #     img = self.random_erasor_I(np.copy(img))
            #     img = self.random_erasor_O(np.copy(img), bboxes)

        # Resize the image to target size and transforms it into a color channel(BGR->RGB),
        # as well as pixel value normalization([0,1])
        img, bboxes = dataAug.Resize((self.img_size, self.img_size), True)(np.copy(img), np.copy(bboxes))
        
        return img, bboxes

if __name__ == "__main__":
    import config.yolov3_config_yoloformat as cfg
    train_dir = f'./custom_data_yolo/train/'
    # dataset = VocDataset(cfg=cfg, anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])   # or  img_size=448
    dataset = YoloDataset(cfg=cfg, data_dir=train_dir, img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
    dataloader = DataLoader(dataset, shuffle=False, batch_size=2, num_workers=0)

    for i, (img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) in enumerate(dataloader):
        # if i==0:
        print(img.shape)
        print(label_sbbox.shape)
        print(label_mbbox.shape)
        print(label_lbbox.shape)
        print(sbboxes.shape)
        print(mbboxes.shape)
        print(lbboxes.shape)

        # if img.shape[0] == 1:
        #     labels = np.concatenate([label_sbbox.reshape(-1, 26), label_mbbox.reshape(-1, 26),
        #                                 label_lbbox.reshape(-1, 26)], axis=0)
        #     labels_mask = labels[..., 4]>0
        #     labels = np.concatenate([labels[labels_mask][..., :4], np.argmax(labels[labels_mask][..., 6:],
        #                             axis=-1).reshape(-1, 1)], axis=-1)

        #     print(labels.shape)
        #     tools.plot_box(labels, img, cfg, id=1)  # now i dont save the images
