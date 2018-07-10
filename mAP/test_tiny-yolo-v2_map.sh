mkdir -p predicted
rm -f predicted/*
cp predicted-yolov2-ncsdk/* predicted/
python mAP.py -na
