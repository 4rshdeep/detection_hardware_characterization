# mean Average Precision scores (mAP)
Get a good idea about the metric [here](https://datascience.stackexchange.com/questions/25119/how-to-calculate-map-for-detection-task-for-the-pascal-voc-challenge)

## TL;DR - How to get the mAP scores on kitti dataset? 

* ssd mobilenet caffe model 
```
./test_caffe_map.sh
1.64% = person AP  
22.56% = vehicle AP  
mAP = 12.10%
```

* ssd mobilenet tensorflow model
``` 
./test_tf_map.sh
1.75% = person AP  
26.92% = vehicle AP  
mAP = 14.34%
```

* tiny yolov2 - neural compute stick
```
./test_tiny-yolo-v2_map.sh
2.41% = person AP  
7.22% = vehicle AP  
mAP = 4.81%
```

## How does this work?
First you need to understand what each folder contains:-
* ground-truth - this folder contains the ground truth about each image of the testset in the format:- `class left top right bottom`
* predicted - before running mAP.py this folder should contain the results from a particular model in the same format
* predicted-caffe-ssd-mobilenet - this folder already contains the results of running the caffe-ssd-mobilenet model on kitti dataset
* predicted-tf-ssd-mobilenet - this folder already contains the results of running the caffe-tf-mobilenet model on kitti dataset
* predicted-yolov2-ncsdk - this folder already contains the results of running the tiny yolo v2 model on kitti dataset
* results - auto generated folder storing the results of the last run of mAP.py
* generate_GTandPredicted - contains scripts to generate ground-truth and predicted text files

## How to use this?

### models used in this repository
To get the mAP value of caffe-ssd-mobilenet/tf-ssd-mobilenet/tiny-yolo-v2 on kitti dataset, you can simply run the appropriate script as mentioned above in the TL;DR section.

### custom models
Follow these steps to get the mAP score of your own custom model on kitti dataset
1. Follow the instructions in `generate_GTandPredicted` folder to generate the ground-truth and predicted text files. 
2. Paste the ground-truth text files for each image in the test dataset inside the ground-truth folder.
3. Paste the predicted text files, again for each image in the predicted folder.
4. Run `python mAP.py`