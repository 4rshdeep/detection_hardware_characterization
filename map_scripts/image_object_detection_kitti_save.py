#! /usr/bin/python3

import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt",help="path to Caffe 'deploy' prototxt file", default="caffe-model/MobileNetSSD_deploy.prototxt")
ap.add_argument("-m", "--model", help="path to Caffe pre-trained model", default="caffe-model/MobileNetSSD_deploy.caffemodel")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
ap.add_argument("-i", "--image", help = "Path to the image", default="img3.png")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
#CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
#   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
#   "sofa", "train", "tvmonitor"]

KITTI_ROOT="/home/arshdeep/datasets/kitti/"
# CLASSES = ["dontcare", "vehicle", "vehicle", "dontcare", "vehicle",
#   "vehicle", "vehicle", "vehicle", "vehicle", "vehicle", "vehicle", "vehicle",
#   "vehicle", "vehicle", "vehicle", "person", "dontcare", "vehicle",
#   "vehicle", "vehicle", "dontcare"]

CLASSES = ["dontcare", "vehicle", "vehicle", "dontcare", "vehicle",
  "dontcare", "vehicle", "vehicle", "dontcare", "vehicle", "dontcare", "dontcare",
  "dontcare", "dontcare", "vehicle", "person", "dontcare", "dontcare",
  "dontcare", "vehicle", "dontcare"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
cwd = os.getcwd()
 
# load our serialized model from disk
#print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


os.chdir(KITTI_ROOT+"training/tmp/")
images=os.listdir()
for img in images:
    frame = cv2.imread(KITTI_ROOT+"training/tmp/"+img)
    # frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    t1 = time.time()
    detections = net.forward()
    t2 = time.time()
   # print("[INFO] Total Inference time is {}".format(t2 - t1))
    # loop over the detections
    f = open(cwd+"/predicted/"+img[0:img.index('.')]+".txt",'w')
    for i in np.arange(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with
    # the prediction
        confidence = detections[0, 0, i, 2]

    # extract the index of the class label from the
    # `detections`, then compute the (x, y)-coordinates of
    # the bounding box for the object

        if confidence>args["confidence"]:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = CLASSES[idx]
            if label!="dontcare":
                f.write(label+' '+str(confidence)+' '+str(startX)+' '+str(startY)+' '+str(endX)+' '+str(endY)+"\n")
                # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    cv2.imwrite(cwd+'/tmpSaved/'+img,frame)
    f.close()
