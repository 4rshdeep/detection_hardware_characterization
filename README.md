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
3. Finding the mAP score
4. Profiling

## Installation 
We need to install some softwares to run the neural nets on a stock embedded device. In this section, we'll layout all the resources that we used to get our systems up and running.

*  Raspberry Pi 3B
	* [Ubuntu MATE 16.04.2 (Xenial)](https://ubuntu-mate.org/raspberry-pi/)
	* [Tensorflow version > 1.4](https://github.com/lhelontra/tensorflow-on-arm)
	* [BVLC Caffe](https://github.com/leo2105/Caffe-installation-Raspberry-Pi-3)
	* OpenCV version 3.4.1 ARSHDEEP
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
  
## Some important notes

## Contributors
 * [Arshdeep Singh](https://github.com/4rshdeep)
 * [Mayank Singh Chauhan](https://github.com/mayanksingh2298)
    
