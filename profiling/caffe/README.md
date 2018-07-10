## Profiling caffe
Unlike tensorflow, caffe doesn't support seemless profiling of models.

We had to make changes in [caffe](https://github.com/weiliu89/caffe/) source and had to build it again. 
We donot want anyone else to feel so hopeless when they try to profile a caffe model.
Hence we document our method here:-

1. Start by cloning [Weiliu's](https://github.com/weiliu89/caffe/) implementaion of caffe.
```
git clone https://github.com/weiliu89/caffe/
cd caffe
cp Makefile.config.example Makefile.config
```
2. Edit `Makefile.config` and `Makefile` according to the system on which you want to build caffe
  * [Jetson TX2](https://jkjung-avt.github.io/caffe-on-tx2/)
  * [Raspberry Pi 3](https://github.com/leo2105/Caffe-installation-Raspberry-Pi-3)
  * [PC](http://installing-caffe-the-right-way.wikidot.com/start)
3. Copy and replace the files from `changed_layers` folder in this repo to `<caffe-root>/src/caffe/`
4. Build caffe
```
make clean
make -j4 all
make -j4 test
make -j4 runtest
make -j4 pycaffe
```
5. Wherever you want to profile caffe code, make sure to export python path as
```
export PYTHONPATH=<caffe-root>/python:$PYTHONPATH
```

## Notes
1. In this repo, we are trying to run `yoloV2Tiny20` and `MobileNetSSD_deploy`, so we only changed the layers:
 * conv_layer.cpp
 * conv_layer.cu
 * batch_norm_layer.cpp
 * batch_norm_layer.cu
 * scale_layer.cpp
 * scale_layer.cu
 * relu_layer.cpp
 * relu_layer.cu
 * pooling_layer.cpp
 * pooling_layer.cu
 * permute_layer.cpp
 * permute_layer.cu
 * flattern_layer.cpp
 * prior_box_layer.cpp
 * concat_layer.cpp
 * concat_layer.cu
 * softmax_layer.cpp
 * softmax_layer.cu
 * detection_output_layer.cpp
 * detection_output_layer.cu
2. We had tried both `ctime` and `chrono` modules. Both of them yield almost same results.
3. Remember to uncomment the line: `caffe.set_mode_gpu()` in the inference script, when running on a gpu.
