# Benchmarking
We used 50 images from the kitti dataset to benchmark the models.

## Caffe
```python
cd caffe
python benchmark_caffe.py
```

## Tensorflow
```python
cd tensorflow
python tf_object_detection.py
```
## Results
| Framework | Architecture | System | Read image(ms) | Infer image(ms) | Visualize image(ms) |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Caffe | SSD-mobilenet | NCS + Computer | 17 | 89 | 0.09 |
| Caffe | SSD-mobilenet | Jetson | 22.4 | 127.8 | 0.6 |
| Caffe | SSD-mobilenet | Raspberry Pi 3 CPU | 90.3 | 550.6 | 2.3 |
| Caffe | SSD-mobilenet | Raspberry Pi 3 Movidius | 54.7 | 125.9 | 1.3 |
| Caffe | SSD-mobilenet | GPU computer | 12.2 | 24.2 | 0.1 |
| Caffe | SSD-mobilenet | Moto G4 Plus | 197 | 1154 | 7.7 |
| Caffe | SSD-mobilenet | Samsung S7 Edge | 9.56 | 200 | 0.33 |
| Tensorflow | SSD-mobilenet | Moto G4 Plus | 83 | 1390 | 2.1 |
| Tensorflow | SSD-mobilenet | Samsung S7 Edge | 2 | 753 | 2.1 |
| Tensorflow | SSD-mobilenet | Jetson | 18 | 219 | ~ 0 |
| Tensorflow | SSD-mobilenet | Raspberry Pi 3 CPU | 60 | 799 | ~ 0 |
| Tensorflow | SSD-mobilenet | GPU computer | 12.4 | 62.5 | ~ 0 |
| Caffe | Tiny-YoloV2 | NCS + Computer | 17 | 174 | ~ 0 |
| Caffe | Tiny-YoloV2 | Jetson | 28 | 123 | 1 |
| Caffe | Tiny-YoloV2 | Raspberry Pi 3 CPU | 99 | 637 | 4 |
| Caffe | Tiny-YoloV2 | Raspberry Pi 3 Movidius | 64 | 215 | 0.22 |
| Caffe | Tiny-YoloV2 | GPU computer | 11 | 16 | ~ 0 |
| Caffe | Tiny-YoloV2 | Moto G4 Plus | 130 | 1364 | -- |
| Caffe | Tiny-YoloV2 | Samsung S7 Edge | 27 | 232 | -- |
| [Darknet]() | YoloV2 | GPU computer | -- | 17 | -- |
| [Darknet](https://pjreddie.com/darknet/) | YoloV2 | Jetson | -- | 119 | -- |







