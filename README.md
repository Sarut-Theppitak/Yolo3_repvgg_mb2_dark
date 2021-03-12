# Yolo3

This repository contains mobilenetv2 repvgg and darknet backbone. But I will firstly explain how to train the repvgg model.

## Training repvgg

### 1

The directory structure for image data should be as follows (please note the test set is in the separate directory and i will explain later)

```
Foo
|-- train
|   |-- images
|   |-- labels
|
|-- valid
|   |-- images
|   |-- labels
|-- valid_results
```

*** your labels should be Yolo format (.txt) file

### 2

Next step you will need to create the weight directory called "./weights". This will be used to store the trained weight as well as pre-trained weight.

I have provided the repvggA1 pre-trained weight trained on VOc for 100 epoch in this link [here](https://drive.google.com/drive/folders/1OMR1sonZI5fc3o8bCsqIAF4O01Gx05IX)

for example your weight directory should look like this:

```
weights
|-- pre_weight_voc_repA1
|   |-- best.pt
```

### 3

Next, you will need to config your path and training parameters in the config file [here](https://github.com/tokyo-ai/RepVGG_based_Yolo3/blob/main/config/yolov3_config_yoloformat.py)

Please change your own validation and test result directory like below

```
EVAL_REUSLTS_DIR = f"C:/Users/theppitak.sarut/Desktop/re_Yolo/custom_data_yolo/valid_results"
TEST_REUSLTS_DIR = f"C:/Users/theppitak.sarut/Desktop/re_Yolo/test_data/test_results"
```

### 4

Then also change the class map EX.

```
DATA = {"CLASSES":['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor'],
        "NUM":20}
```

or

```
DATA = {"CLASSES":['cancer'],
        "NUM":1}
```

### 5

You can set the training parameteers as you want. Below is the defualt I used.

```
TRAIN = {
         "TRAIN_IMG_SIZE":384,
         "AUGMENT":True,
        #  "BATCH_SIZE":8,
         "BATCH_SIZE":4,
         "MULTI_SCALE_TRAIN":True, # If True then TRAIN_IMG_SIZE almost has no meaning it will use 256 only for 10 eporch
         "IOU_THRESHOLD_LOSS":0.5,
        #  "EPOCHS":300,
         "EPOCHS":300,
         "NUMBER_WORKERS":0,
         "MOMENTUM":0.9,
         "WEIGHT_DECAY":0.001,  #0.0005
         "LR_INIT":1e-4,
         "LR_END":1e-7,
         "WARMUP_EPOCHS":2  # or 0  2
         } 
 ```

### 6

Please also make a blank directory called "./logs" also to save the tensorbroad file

Now yoou should be able to start the training script

## Training command

You can change the argument in the train.py directly (line 318-322)

```
parser.add_argument('--weight_path', type=str, default=f'./weight/pre_weight_voc_repA1/best.pt', help='weight file path to load') # use  '' to train from scarch
parser.add_argument('--resume', action='store_true',default=False,  help='resume training flag')
parser.add_argument('--train_dir', type=str, default=f'./custom_data_yolo/train/', help='train directory')
parser.add_argument('--valid_dir', type=str, default=f'./custom_data_yolo/valid/', help='valid directory')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
```

or use the training command like this

```
python train.py --weight_path './weight/pre_weight_voc_repA1/best.pt' --train_dir ./custom_data_yolo/train/ --valid_dir ./custom_data_yolo/valid/
```

Please note that --weight_path is the pre-trained weight path. It can be any location. However, I asked you to create "/.weight" because I set the training scrpit to save the being trained weight in "/.weight" directory

## Change the model architecture

You can change the model backbone to any type of repvgg. please see the line 49-52 of ./model/yolo3.py [here](https://github.com/tokyo-ai/RepVGG_based_Yolo3/blob/main/model/yolov3.py)

(note you will have to change the fileters_in of FPN_YOLOV3 class) >>> standard [512, 258, 128]

For instance: self.w_A1 = [1, 1, 1, 2.5] >> fileters_in=[1280, 256, 128] self.w_A0 = w_A0 = [0.75, 0.75, 0.75, 2.5] >> fileters_in=[1280, 192, 96]

Ex. repvggA1

```
self.__backbone = RepVGG(num_blocks=self.A_num_blocks, width_multiplier=self.w_A1, 
                      override_groups_map=None, deploy=deploy)
self.__fpn = FPN_YOLOV3(fileters_in=[1280, 256, 128],
                     fileters_out=[self.__out_channel, self.__out_channel, self.__out_channel])
```

Ex. repvggA1g4 (we have upto g2-g32)

```
self.__backbone = RepVGG(num_blocks=self.A_num_blocks, width_multiplier=self.w_A1, 
                      override_groups_map=self.g4_map, deploy=deploy)
self.__fpn = FPN_YOLOV3(fileters_in=[1280, 256, 128],
                     fileters_out=[self.__out_channel, self.__out_channel, self.__out_channel])
```

Ex. repvggA0

```
self.__backbone = RepVGG(num_blocks=self.A_num_blocks, width_multiplier=self.w_A0, 
                      override_groups_map=None, deploy=deploy)
self.__fpn = FPN_YOLOV3(fileters_in=[1280, 192, 96],
                     fileters_out=[self.__out_channel, self.__out_channel, self.__out_channel])
Traing using Mobilev2 backbone
```

## Traing using Mobilev2 backbone

You can comment out the line line 49-52 and uncommnet the line 42-46

## Testing the model

You will use the test.py with arguments below for example:

```
parser.add_argument('--use_weight_path', type=str, default='./weight/repA1g4_seed1_10each_0.001decay/best.pt', help='weight file path')
parser.add_argument('--img_test_dir', type=str, default='./test_data/test/images', help='test folder containing the images for test')
parser.add_argument('--annotation_dir', type=str, default='./test_data/test/labels', help='annotation path for mAP or '' or None ex. ./test_data/test/labels')
```

You will have to create a directory to contain the test data like below:

```
Foo
|-- test
|   |-- images
|   |-- labels
|
|-- test_results
```

You dont have to have test_results directory but I used it because I specify the test result directtory in the config file "config/yolov3_config_yoloformat.py"

```
TEST_REUSLTS_DIR = f"C:/Users/theppitak.sarut/Desktop/re_Yolo/test_data/test_results"
```

You can set the training parameteers as you want. Below is the defualt I used in the config file.

```
TEST = {
        "TEST_IMG_SIZE":384,
        "BATCH_SIZE":1,   #Fix
        "NUMBER_WORKERS":0, # Fix
        "CONF_THRESH":0.01,  # or 0.5 
        "NMS_THRESH":0.5,   # IOU thrs during nms
        "MULTI_SCALE_TEST":False, # Fix
        "FLIP_TEST":False,
        "SHOW_RESULT": False,
        "SAVE_RESULT": True
        }
```

## Warning

You should change achitecture in yolo3.py to match with the model weight architecture you want to test.

## Converting to deploy model

Please use the convert_repvgg.py to convert your training-time model to deploy_time model. please use the line 6-45.

After that you can test the eqilvalence of the 2 models by using line 50-71.

## Warning

You should change achitecture in yolo3.py to match with the model weight architecture you want to convert.
