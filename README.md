# Object detection hardware characterization
  This repository hosts the code and the models used for analyzing various models on different embedded devices. 
  
## Overview
  For our work we used 5 different hardwares,
 * NVIDIA Jetson TX2 Development Kit
 * GPU (NVIDIA GeForce GTX 1070)
 * Intel® Movidius™ Neural Compute Stick
 * Raspberry Pi 3B 
 * Mobile Phone (Samsung Galaxy S7 edge G935D)

and 3 different Object Detection Models 
 * Caffe implementation of [Single Shot Detector](https://arxiv.org/abs/1512.02325) with Mobilenet as a feature extractor.
 * Tensorflow implementation of [Single Shot Detector](https://arxiv.org/abs/1512.02325) with Mobilenet as a feature extractor.
 * Caffe implementation of [Tiny-YoloV2](https://pjreddie.com/media/files/papers/YOLO9000.pdf) 
