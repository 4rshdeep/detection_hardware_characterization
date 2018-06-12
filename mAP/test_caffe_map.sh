mkdir -p predicted
rm -f predicted/*
cp predicted-caffe-ssd-mobilenet/* predicted/
python mAP.py -na

