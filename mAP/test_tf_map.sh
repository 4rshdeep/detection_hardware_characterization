mkdir -p predicted
rm -f predicted/*
cp predicted-tf-ssd-mobilenet/* predicted/
python mAP.py -na
