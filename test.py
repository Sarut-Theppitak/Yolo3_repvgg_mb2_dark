from torch.utils.data import DataLoader
import utils.gpu as gpu
from model.yolov3 import Yolov3
from tqdm import tqdm
from utils.tools import *
from eval.evaluator import YoloEvaluator
import argparse
import os
import shutil
from utils.visualize import *


class Tester(object):
    def __init__(self,
                 cfg,
                 use_weight_path,
                 img_dir,
                 gpu_id=0,
                 img_size=488,
                 annotation_dir='',
                 ):
        self.cfg = cfg
        self.img_size = img_size
        self.__num_class = cfg.DATA["NUM"]
        self.__conf_threshold = cfg.TEST["CONF_THRESH"]
        self.__nms_threshold = cfg.TEST["NMS_THRESH"]
        self.__visual_threshold = cfg.TEST["VISUAL_THRESH"]
        self.__device = gpu.select_device(gpu_id)
        self.__multi_scale_test = cfg.TEST["MULTI_SCALE_TEST"]
        self.__flip_test = cfg.TEST["FLIP_TEST"]

        self.__img_dir = img_dir
        self.__annotation_dir = annotation_dir

        self.__classes = cfg.DATA["CLASSES"]

        self.__model = Yolov3(cfg).to(self.__device)

        self.__load_model_weights(use_weight_path)

        self.__evalter = YoloEvaluator(self.__model, cfg, img_dir, mode="test", anno_dir=annotation_dir)

        self.__visual_imgs = 0

    def __load_model_weights(self, weight_path):
        print("loading weight file from : {}".format(weight_path))

        weight = os.path.join(weight_path)
        chkpt = torch.load(weight, map_location=self.__device)
        # print(self.__model._Yolov3__backnone.conv1.weight[0,0,0])
        self.__model.load_state_dict(chkpt)    # chkpt['model'] when use full check point
        print("loading weight file is done")
        del chkpt
        # print(self.__model._Yolov3__backnone.conv1.weight[0,0,0])
        
    def test(self):

        result_path =  self.cfg.TEST_REUSLTS_DIR
        pred_cachedir = os.path.join(result_path, "cache")

        if os.path.exists(pred_cachedir):
            shutil.rmtree(pred_cachedir)  # delete the cache directory
        os.mkdir(pred_cachedir)

        imgs = os.listdir(self.__img_dir)
        for v in imgs:
            path = os.path.join(self.__img_dir, v)
            print("test images : {}".format(path))

            img = cv2.imread(path)
            if img is None:
                print(f'****Could not read an image****: {path}')
                print('****Continue to the next image****')
                continue

            bboxes_prd = self.__evalter.get_bbox(img, self.__multi_scale_test, self.__flip_test)     # (xmin, ymin, xmax, ymax, score, class) compatible with the original image size.  it will resize the image in the __predict of the Evaluator 
            # **** we can select the class to visualize also it is in the mod_evaluator.py  bboxes_prd can be just np.array([])
            # There are still some boxes that > score_thres even after nmf if we set score_thres too low (ex.0.01) However, it will be remove in the visualize_boxes()
            if bboxes_prd.shape[0] != 0:
                boxes = bboxes_prd[..., :4]
                class_inds = bboxes_prd[..., 5].astype(np.int32)
                scores = bboxes_prd[..., 4]
                # this will not draw the boxes that have confidence score < 0.5  >> fixed 
                visualize_boxes(image=img, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.__classes, draw_thes=self.__visual_threshold)

            if self.__visual_imgs < 110:               
                if cfg.TEST['SHOW_RESULT']:
                    cv2.imshow('test_result', img) 
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                if cfg.TEST['SAVE_RESULT']:
                    save_file = os.path.join(result_path, "{}".format(v))
                    cv2.imwrite(save_file, img)
                    print("saved images : {}".format(save_file))

                self.__visual_imgs += 1

            # write predict result in the text files
            for bbox in bboxes_prd:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])

                class_name =  self.__classes[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = map(str, coor)
                s = ' '.join([v, score, xmin, ymin, xmax, ymax]) + '\n'

                with open(os.path.join(pred_cachedir, 'det_result_' + class_name + '.txt'), 'a') as f:
                    f.write(s)

        print("Finished infernce in testing")
        if self.__annotation_dir:
            mAP = 0
            print('*' * 20 + "Validate mAP" + '*' * 20)
            with torch.no_grad():
                APs = self.__evalter.APs_run(self.__multi_scale_test, self.__flip_test)
                valid_ap = 0
                for cla in APs:
                    print("{} --> mAP : {}".format(cla, APs[cla]))
                    if (APs[cla] is not None) and (~np.isnan(APs[cla])) :   # actually nan should be 0 value
                        mAP += APs[cla]
                        valid_ap += 1
                mAP = mAP / valid_ap
                print('mAP:%g' % (mAP))

        else:
            print('No mAP valulation, No annotation path')

if __name__ == "__main__":
    import config.yolov3_config_yoloformat as cfg
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_weight_path', type=str, default='./weight/mobile2_0.0005_384/best.pt', help='weight file path')
    parser.add_argument('--img_test_dir', type=str, default='./test_data/test/images', help='test folder containing the images for test')
    parser.add_argument('--annotation_dir', type=str, default='./test_data/test/labels', help='annotation path for mAP or '' or None ex. ./test_data/test/labels')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    opt = parser.parse_args()

    Tester( cfg=cfg,
            use_weight_path=opt.use_weight_path,
            img_dir=opt.img_test_dir,
            gpu_id=opt.gpu_id,
            annotation_dir=opt.annotation_dir,
            img_size=cfg.TEST['TEST_IMG_SIZE']).test()
