# Find mAP of running a custom model on kitti dataset
## Get the ground truth text files
1. 
```python
python from_kitti.py -p PATH_TO_DATASET_LABELS
```

## Get predicted text files - caffe
First correct the kitti-root on line 26
```python 
python image_object_detection_kitti.py -p PATH_TO_PROTOTXT -m PATH_TO_MODEL
```

## Get predicted text files - tensorflow
1. Correct the kitti-root on line 64
2. Change the lines 31-38 according to your model.
```python 
python tf_object_detection.py
```

## Note
These scripts convert the object classes into `person` and `vehicle`