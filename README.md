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
3. Finding the [mAP](https://stackoverflow.com/a/37498432)) score
4. Profiling
5. Energy values

## Installation 
We need to install some softwares to run the neural nets on a stock embedded device. In this section, we'll layout all the resources that we used to get our systems up and running.

* Raspberry Pi 3B (running Ubuntu MATE 16.04.2)
	* [Ubuntu MATE 16.04.2 (Xenial)](https://ubuntu-mate.org/raspberry-pi/)
	* [Tensorflow version > 1.4](https://github.com/lhelontra/tensorflow-on-arm)
	* [BVLC Caffe](https://github.com/leo2105/Caffe-installation-Raspberry-Pi-3)
	* [OpenCV version 3.4.1](https://www.pyimagesearch.com/2017/10/09/optimizing-opencv-on-the-raspberry-pi/)
	* [Google Protocol Buffers Compiler ](http://osdevlab.blogspot.com/2016/03/how-to-install-google-protocol-buffers.html)
	* [API for Intel Movidius Stick](https://www.pyimagesearch.com/2018/02/12/getting-started-with-the-intel-movidius-neural-compute-stick/)
* NVIDIA Jetson TX2 Development Kit (running L4T 28.2.1)
	* [NVIDIA JetPack SDK Version 3.2](https://developer.nvidia.com/embedded/jetpack)
	* [BVLC Caffe](https://jkjung-avt.github.io/caffe-on-tx2/) 
	* [Tensorflow version > 1.4](https://github.com/peterlee0127/tensorflow-nvJetson/releases)
	* [OpenCV version 3.4.1](https://jkjung-avt.github.io/opencv3-on-tx2/) built from sources

* GPU (NVIDIA GeForce GTX 1070)
	* [BVLC Caffe](https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide)
	* [Tensorflow version > 1.4](https://www.tensorflow.org/install/install_linux)
	* [Opencv version 3.4.1](https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/) 
	* [Intel® Movidius™ Neural Compute SDK](https://movidius.github.io/ncsdk/install.html)

* Android
	* [Caffe models](https://docs.opencv.org/3.4.0/d0/d6c/tutorial_dnn_android.html)
	* [Tensorflow models](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android)
	
## Benchmarking
We have benchmarked the models on different systems and calculated the time taken to `read the image`, `infer the image` and `visualize the image`. The results along with the scripts are documeneted [here](https://github.com/4rshdeep/detection_hardware_characterization/tree/master/benchmark).

## Finding the mAP score
The details about the mean average precision scores of different models can be found [here](https://github.com/4rshdeep/detection_hardware_characterization/tree/master/mAP).

## Profiling
We have also tried to find the time taken to run each layer of the neural networks on different systems. The results along with the scripts can be found [here](https://github.com/4rshdeep/detection_hardware_characterization/tree/master/profiling).

## Energy Values
A thorough study of embedded systems is incomplete if we donot provide data about the power and current consumption of different systems. We used [this](https://www.amazon.in/Digital-Monitor-Energy-Tester-14014816MG/dp/B01FW9D7SK) device to measure the current and power consumed by different systems as they were inferencing images. ARSHDEEP (add the link to the sheet)

  
## Some important notes
* We only need to install the `Intel® Movidius™ Neural Compute Stick API` on embedded systems to run a graph compiled for the neural compute stick. The complete `SDK` is installed on a computer where we can compile, profile and tune neural networks to run on a neural compute stick.
* We chose to work with tiny yolo because the original yolo won't run on a Raspberry Pi CPU. The RAM was too small.
* ssd chalane ko ek alag branch se install karna hota hai. ARSHDEEP handle this
* The jetson can be overclocked to further increase its computation power. Unless otherwise stated, the overclocked mode was used to run the networks. ARSHDEEP add how to overclock
* Neural Compute Stick doesn't yet support the tensorflow version of SSD-Mobilenet as of June 2018. [Reference](https://ncsforum.movidius.com/discussion/667/tensorflow-ssd-mobilenet)
* We weren't able to profile and find energy values for android device.

## Contributors
* [Arshdeep Singh](https://github.com/4rshdeep)
* [Mayank Singh Chauhan](https://github.com/mayanksingh2298)

## Credits
* For computing [mAP](https://github.com/Cartucho/mAP)
    
