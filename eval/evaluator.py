import os
import shutil
import cv2
import numpy as np
from utils.data_augment import Resize
import torch
from utils.tools import xywh2xyxy, nms
from tqdm import tqdm

class BaseEval():
    def get_bbox(self, img, multi_test=False, flip_test=False):
        if multi_test:
            test_input_sizes = range(320, 640, 96)
            bboxes_list = []
            for test_input_size in test_input_sizes:
                valid_scale =(0, np.inf)
                bboxes_list.append(self.__predict(img, test_input_size, valid_scale))
                if flip_test:
                    bboxes_flip = self.__predict(img[:, ::-1], test_input_size, valid_scale)
                    bboxes_flip[:, [0, 2]] = img.shape[1] - bboxes_flip[:, [2, 0]]
                    bboxes_list.append(bboxes_flip)
            bboxes = np.row_stack(bboxes_list)
        else:
            bboxes = self.__predict(img, self.val_shape, (0, np.inf))
        # boxes  (xmin, ymin, xmax, ymax, score, class)

        ################################## In case we want to visualize only the selected classes ################
        # selected_classes = ['car', 'dog']
        # selected_idxs = [ self.cfg.DATA['CLASSES'].index(clss) for clss in selected_classes]       
        # mask = np.in1d(bboxes[:, 5], selected_idxs)
        # bboxes = bboxes[mask]
        ###########################################################################################################

        bboxes = nms(bboxes, self.conf_thresh, self.nms_thresh)   # why still self.conf_thresh? it already filter in the self.__predict

        return bboxes

    def __predict(self, img, test_shape, valid_scale):
        org_img = np.copy(img)
        org_h, org_w, _ = org_img.shape

        img = self.__get_img_tensor(img, test_shape).to(self.device) # return another img objec, the original img object will not be changes
        self.model.eval()
        with torch.no_grad():
            _, p_d = self.model(img)    # p_d is the [bs, all boxes in 3 yolo heads, 8]   when __prodict the bs = 1
        pred_bbox = p_d.squeeze().cpu().numpy() # now >> [all boxes , 8]   boxes are in xywh in pixel unit
        bboxes = self.__convert_pred(pred_bbox, test_shape, (org_h, org_w), valid_scale)

        return bboxes

    def __get_img_tensor(self, img, test_shape):
        img = Resize((test_shape, test_shape), correct_box=False)(img, None).transpose(2, 0, 1)   # resize image + normalize from 0range [0,1]
        return torch.from_numpy(img[np.newaxis, ...]).float()


    def __convert_pred(self, pred_bbox, test_input_size, org_img_shape, valid_scale):
        """
        The prediction box is filtered to remove the unreasonable scale
        """
        # pred_bbox is numpy in xywh in pixel unit and since it is the decode version without filtering, there are some boxesthat goes beyond the wdith and the hight of the images
        pred_coor = xywh2xyxy(pred_bbox[:, :4])
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # (1)
        # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
        # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
        # transalte >> It should be noted that no matter what data enhancement method we use during training, it will not affect the conversion method here.
        # transalte >> Suppose we use conversion method A for the input test image, then the conversion method for bbox here is the reverse process of method A
        # comclusion. it is just the reverse of the resize
        org_h, org_w = org_img_shape
        resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
        dw = (test_input_size - resize_ratio * org_w) / 2   #Pai its the pad value needed after the resize. because we resize first and then padd
        dh = (test_input_size - resize_ratio * org_h) / 2
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio   # at this line the box is in the pixez scale but compatible with the original image size

        # (2)将预测的bbox中超出原图的部分裁掉 Cut out the part of the predicted bbox that exceeds the original image
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
        # (3)将无效bbox的coor置为0 Set the coor of the invalid bbox to 0   the boxes is in xyxy    for example   x1 should be smaller than x2
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0  # set those invalid boxes to [0, 0, 0, 0]

        # (4)去掉不在有效范围内的bbox 去掉不在有效范围内的 bbox Remove the bbox that is not in the valid range    
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))   # bboxes_scale = area of the box scqre root  (x2-x1) * (y2-y1)
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))  # bboxes_scale < valid_scale[1] because there is some nan value(from the last step) ex. np.sqrt(-2)  

        # (5)将score低于score_threshold的bbox去掉
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > self.conf_thresh

        mask = np.logical_and(scale_mask, score_mask)

        coors = pred_coor[mask]
        scores = scores[mask]
        classes = classes[mask]

        bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)  # shape = [all valid boxes, 6]  first 4 is boxes xyxy , then score , then classes (from argmax so from range [0, num_clss])

        return bboxes   # the boxes are the pixez scale but compatible with the original image size and already filter all valid boxes (> confidence threshold and aslo weired boxes)

class YoloEvaluator(BaseEval):
    def __init__(self, model, cfg, img_dir, mode, anno_dir=None):  # model are 'test' or 'eval'
        self.mode = mode
        if mode == 'test':
            mode_cfg = cfg.TEST
        elif mode == 'eval':
            mode_cfg = cfg.EVAL

        self.classes = cfg.DATA["CLASSES"]
        self.conf_thresh = mode_cfg["CONF_THRESH"]
        self.nms_thresh = mode_cfg["NMS_THRESH"]
        self.val_shape =  mode_cfg["TEST_IMG_SIZE"]
        self.cfg = cfg

        self.img_dir = img_dir   
        self.anno_dir = anno_dir  
        self.__visual_imgs = 0

        self.model = model
        self.device = next(model.parameters()).device

    def APs_run(self, multi_test=False, flip_test=False):
        img_names = os.listdir(self.img_dir)
        
        anno_recs = {} 
        for img_name in img_names:
            anno_path = os.path.join(self.anno_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
            try:
                boxes = np.loadtxt(anno_path).reshape(-1, 5)
                # Here we may need to add filters > no i will not because it yolo3
                anno_recs[img_name] = boxes
            except Exception as e:
                anno_recs[img_name] = np.array([])
                print(f'system error massage: {e}')
                print(f"Annotation file is not existed (mAP): '{anno_path}'.")
    
        ######  start to calulate the mAP
        if self.mode == "test":
            cache_path =  os.path.join(self.cfg.TEST_REUSLTS_DIR, "cache")
        elif self.mode == "eval":
            cache_path =  os.path.join(self.cfg.EVAL_REUSLTS_DIR, "cache")
        else:
            assert False , "The mode is invalid"

        det_files = os.path.join(cache_path, 'det_result_{}.txt')
        # class_to_id = dict(zip(self.classes, range(self.num_classes)))

        APs = {}
        for cls_idx, cla in enumerate(self.classes):
            det_file = det_files.format(cla)
            R, P, AP = self.__eval_ap(det_file, img_names, anno_recs, cls_idx)
            APs[cla] = AP
            
        return APs

    def __eval_ap(self, det_file, img_names, anno_recs, cls_idx, iou_thresh=0.5, use_07_metric=False):
        # extract gt objects for this class
        class_recs = {}
        npos = 0   #number of all gound truth boxes from all images for this class (number positive)
        for imagename in img_names:
            img_path = os.path.join(self.img_dir, imagename)
            img = cv2.imread(img_path )
            pixel_h, pixel_w = img.shape[0], img.shape[1]

            R = [bbox[1:] for bbox in anno_recs[imagename] if bbox[0] == cls_idx]   # all bboxes in test set relative to a class in one images
            bboxs = np.array(R)   # all boxes in test set relative to a class in one images
            if len(R) > 0:
                bboxs[:, ::2] = (bboxs[:, ::2] * pixel_w)
                bboxs[:, 1::2] = (bboxs[:, 1::2] * pixel_h)
                bboxs = xywh2xyxy(bboxs).astype('int')  # because yolo format is in xywh
                # # for debugging 
                # im = np.copy(img)
                # for box in bboxs:
                #     cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]) , (0,255,0), 2)
                #     font = cv2.FONT_HERSHEY_SIMPLEX
                #     color = (255, 0, 0) 
                # cv2.imshow('test', im) 
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

            det = [False] * len(R)    # for keeping track
            npos = npos + len(R)
            class_recs[imagename] = {'bbox': bboxs,
                                    'det': det}

        # Finising extracting the annotation to this class

        # Check if we have any detection from all images on this class
        if not os.path.isfile(det_file):
            if npos == 0:
                return None, None, None    # because we donot care
            else:
                return np.nan, np.nan, np.nan    # it make tp = 0  so  pre = recall = 0
        if npos == 0:
            return np.nan, np.nan, np.nan     # same it makes tp = 0  so  pre = recall = 0

        # Read detectiions for this class
        with open(det_file, 'r') as f:
            lines = f.readlines()
        splitlines = [x.strip().split(' ') for x in lines]

        det_image_names = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        det_image_names = [det_image_names[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(det_image_names)    # num detection
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[det_image_names[d]]   # gt
            bb = BB[d, :].astype(float)          # detection
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)       # gt

            if BBGT.size > 0:  # if there is bbox
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                    (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                    (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > iou_thresh:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        AP = self.__cal_ap_from_pre_rec(rec, prec, use_07_metric)
        print(f'Recall: {rec[-1]}, Precision: {prec[-1]}')
        return rec, prec, AP

    def __cal_ap_from_pre_rec(self, rec, prec, use_07_metric=False):
        """ ap = cal_ap_from_pre_rec(rec, prec, [use_07_metric])
        Compute AP given precision and recall.
        If use_07_metric is true, uses the
       07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap