
* run ./test_caffe_map.sh to get map scores for map scores of ssd mobilenet caffe model 
```
./test_caffe_map
1.64% = person AP  
22.56% = vehicle AP  
mAP = 12.10%
```


* run ./test_tf_map.sh to get map scores for map scores of ssd mobilenet tensorflow model
``` 
./test_tf_map.sh
1.75% = person AP  
26.92% = vehicle AP  
mAP = 14.34%
```

* for tiny yolov2
```
 python mAP.py -na                                                       2 ↵
2.41% = person AP  
7.22% = vehicle AP  
mAP = 4.81%
```