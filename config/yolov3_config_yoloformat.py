EVAL_REUSLTS_DIR = f"C:/Users/theppitak.sarut/Desktop/re_Yolo/custom_data_yolo/valid_results"
TEST_REUSLTS_DIR = f"C:/Users/theppitak.sarut/Desktop/re_Yolo/test_data/test_results"

# DATA = {"CLASSES":['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
#            'train', 'tvmonitor'],
#         "NUM":20}


# DATA = {"CLASSES":['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
#         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
#         'hair drier', 'toothbrush'],
#         "NUM":80}

DATA = {"CLASSES":['cancer'],
        "NUM":1}

# model
MODEL = {"ANCHORS":[[(1.075, 1.392), (1.725, 3.2), (3.535, 2.465)],  # Anchors for small obj (1/8) >> most number of cells
            [(1.6075, 3.2675), (3.3225, 2.41), (3.16, 6.375)],  # Anchors for medium obj
            [(3.1075, 2.41), (4.1775, 5.3025), (9.991, 8.7325)]] ,# Anchors for big obj  # all anchors is in the grid scale of it yolo layers
         "STRIDES":[8, 16, 32],
         "ANCHORS_PER_SCLAE":3
         }  # new_value = 1.25 * (new_size / 448)

# train
TRAIN = {
         "TRAIN_IMG_SIZE":384,
         "AUGMENT":True,
        #  "BATCH_SIZE":8,
         "BATCH_SIZE":4,
         "MULTI_SCALE_TRAIN":False, # If True then TRAIN_IMG_SIZE almost has no meaning it will use defalut only for 10 eporch
         "IOU_THRESHOLD_LOSS":0.5,
        #  "EPOCHS":300,
         "EPOCHS":300,
         "NUMBER_WORKERS":0,
         "MOMENTUM":0.9,
         "WEIGHT_DECAY":0.0005,  #0.0005
         "LR_INIT":1e-4,
         "LR_END":1e-7,
         "WARMUP_EPOCHS":2  # or 0  2
         } 

# eval
EVAL = {
        "TEST_IMG_SIZE":384,
        "BATCH_SIZE":1,   
        "NUMBER_WORKERS":0, 
        "CONF_THRESH":0.01, 
        "NMS_THRESH":0.5,     # IOU thrs during nms
        "MULTI_SCALE_TEST":False,
        "FLIP_TEST":False,
        "SHOW_RESULT": False,
        "SAVE_RESULT": True
        }

# test
TEST = {
        "TEST_IMG_SIZE":384,
        "BATCH_SIZE":1,   #Fix
        "NUMBER_WORKERS":0, # Fix
        "CONF_THRESH":0.5, 
        "VISUAL_THRESH":0.5,
        "NMS_THRESH":0.5,   # IOU thrs during nms
        "MULTI_SCALE_TEST":False, # Fix
        "FLIP_TEST":False,
        "SHOW_RESULT": False,
        "SAVE_RESULT": True
        }