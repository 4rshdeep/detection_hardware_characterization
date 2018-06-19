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
  * [Jetson TX2] (https://jkjung-avt.github.io/caffe-on-tx2/)
  * [Raspberry Pi 3](https://github.com/leo2105/Caffe-installation-Raspberry-Pi-3)
  * [PC] (http://installing-caffe-the-right-way.wikidot.com/start)
3. Copy and replace the `layers` folder from this repo to `<caffe-root>/src/caffe/>
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
