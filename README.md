

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


## Installation 
Software requirements for different hardwares
*  NVIDIA Jetson TX2 Development Kit 
    * NVIDIA JetPack SDK Version 3.2 
    * BVLC Caffe 
    * Tensorflow version > 1.4
    * OpenCV version 3.4.1 built from sources

* GPU (NVIDIA GeForce GTX 1070)
    *   BVLC Caffe
    *   Tensorflow version > 1.4
    *   Opencv version 3.4.1 built from sources
    *    Intel® Movidius™ Neural Compute SDK for profiling, tuning and compiling trained models for NCS.
    
* Intel® Movidius™ Neural Compute Stick needs a host computer for inference which needs to have Intel® Movidius™ Neural Compute API installed and the stick does not have any other requirements.

*  Raspberry Pi 3B
	*  Ubuntu MATE 16.04.2 (Xenial)
	* Tensorflow version > 1.4
	* BVLC Caffe
	* OpenCV version 3.4.1 
	* [Google Protocol Buffers Compiler ](http://osdevlab.blogspot.com/2016/03/how-to-install-google-protocol-buffers.html)
	
	# TODO @mayank * Mobile Phone
  
 

    
