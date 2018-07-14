# Realtime Detection using Deep Learning on Embedded Platforms
## Motivation
Processing high definition videos at powerful GPU servers might give the best performance in road traffic monitoring. But the poor broadband infrastructure in developing countries might prohibit real time streaming of HD videos from roads to servers. In-situ processing might mandate using mobile and embedded platforms, but their processor and battery constraints can conflict with heavy computation and low latency requirements.

That is why we wanted to see whether we can use [embedded systems](https://en.wikipedia.org/wiki/Embedded_system) for inferencing. 

This work was done by Mayank Singh Chauhan and Arshdeep Singh, after our second year of Computer Science at [Indian Institute of Technology, Delhi](http://www.iitd.ac.in/) under the guidance of Professor [Rijurekha Sen](http://www.cse.iitd.ernet.in/~rijurekha/) as part of [Summer Undergraduate Research Award (SURA)](http://ird.iitd.ac.in/content/summer-undergraduate-research-award-sura) in the summer break of academic year 2017-18

## Introduction
We obtained pre-trained object detection models and used them for inferencing on different embedded systems. The following system setups were tested:
* [Raspberry Pi 3B](https://www.raspberrypi.org/products/raspberry-pi-3-model-b/) - A low cost fully functional system.
* Raspberry Pi 3B + [Intel® Movidius™ Neural Compute Stick](https://developer.movidius.com/) - NCS is a tiny fanless deep learning device that helps in inferencing and tuning neural networks.
* [NVIDIA Jetson TX2 Development Kit](https://developer.nvidia.com/embedded/buy/jetson-tx2-devkit) - A costly but highly efficient embedded system.
* Mobile Phone (Samsung Galaxy S7 edge G935D and Motorolla Moto G4 Plus) - Android devices are the most common embedded systems that one can see.
* GPU (NVIDIA GeForce GTX 1070) - Just for comparison, we also run the networks on a GPU enabled computer system.

The following three object detection models were tested:-
* Caffe implementation of [Single Shot Detector](https://arxiv.org/abs/1512.02325) with Mobilenet as a feature extractor.
* Tensorflow implementation of [Single Shot Detector](https://arxiv.org/abs/1512.02325) with Mobilenet as a feature extractor.
* Caffe implementation of [Tiny-YoloV2](https://pjreddie.com/media/files/papers/YOLO9000.pdf).

We'll talk about the following in order:-
1. Installation
2. Benchmarking
3. Finding the [mAP](https://stackoverflow.com/a/37498432) score
4. Profiling

## Installation 
We need to install some softwares to run the neural nets on a stock embedded device. In this section, we'll layout all the resources that we used to get our systems up and running.

*  Raspberry Pi 3B
	* [Ubuntu MATE 16.04.2 (Xenial)](https://ubuntu-mate.org/raspberry-pi/)
	* [Tensorflow version > 1.4](https://github.com/lhelontra/tensorflow-on-arm)
	* [BVLC Caffe](https://github.com/leo2105/Caffe-installation-Raspberry-Pi-3)
	* [OpenCV version 3.4.1](https://www.pyimagesearch.com/2017/10/09/optimizing-opencv-on-the-raspberry-pi/)
	* [Google Protocol Buffers Compiler ](http://osdevlab.blogspot.com/2016/03/how-to-install-google-protocol-buffers.html)
	* [API for Intel Movidius Stick](https://www.pyimagesearch.com/2018/02/12/getting-started-with-the-intel-movidius-neural-compute-stick/)
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

	
	# TODO @mayank * Mobile Phone

## Benchmarking

## Finding the [mAP](https://stackoverflow.com/a/37498432) score
In practice, a higher mAP value indicates a better performance, given your ground-truth and set of classes. We evaluate the performance using the mAP criterium defined in the [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) competition. 
It involves first calculating the Average Precision(AP) for each of the class present in the ground-tructh and then taking mean of all the APs to get the mAP value.

### How to calculate mAP score?
* Create the ground-truth files
* Move the ground-truth files into the folder ground-truth/
* Create the predicted objects files
* Move the predictions files into the folder predicted/
* Run the code: python mAP.py or python mAP.py -na for running without any animationand YOLO files into the required format.

### Creating ground-truth files
* Create a separate ground-truth text file for each image.
* Use matching names (e.g. image: "image_1.jpg", ground-truth: "image_1.txt"; "image_2.jpg", "image_2.txt"...).
* In these files, each line should be in the following format:
```<class_name> <left> <top> <right> <bottom>```
* E.g. "image_1.txt":
```
tvmonitor 2 10 173 238
book 439 157 556 241
book 437 246 518 351
pottedplant 272 190 316 259
```
### Creating predicted objects files
* Create a separate predicted objects text file for each image.
* Use matching names (e.g. image: "image_1.jpg", predicted: "image_1.txt"; "image_2.jpg", "image_2.txt"...).
* In these files, each line should be in the following format:
```<class_name> <confidence> <left> <top> <right> <bottom>```
* E.g. "image_1.txt":
```
tvmonitor 0.471781 0 13 174 244
cup 0.414941 274 226 301 265
book 0.460851 429 219 528 247
chair 0.292345 0 199 88 436
book 0.269833 433 260 506 336
```

## Profiling

## Some important notes

## Credits
For computing [mAP](https://github.com/Cartucho/mAP) 

## Contributors
 * [Arshdeep Singh](https://github.com/4rshdeep)
 * [Mayank Singh Chauhan](https://github.com/mayanksingh2298)
    
