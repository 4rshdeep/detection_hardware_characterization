import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default="../../models/caffe/MobileNetSSD_deploy.prototxt",
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="../../models/caffe/MobileNetSSD_deploy.caffemodel",
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
ap.add_argument("-i", "--image", help = "Path to the image directory", default="/../images/")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
 
# load our serialized model from disk
print("[INFO] loading model...")

t1 = time.time()
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
t2 = time.time()
print("[INFO] Time taken in loading Model is {}".format(t2-t1))


cwd = os.getcwd()
dir = args["image"]
dir = cwd+dir
files = os.listdir(dir)
# print(files)
# print(dir)
# exit()

total_images    = len(files)
loading_image   = 0
inference_image = 0
vis_image       = 0

############ Random first pass ###################
path = dir+files[0]
frame = cv2.imread(path)
(h, w) = frame.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
    0.007843, (300, 300), 127.5)
net.setInput(blob)
detections = net.forward() 
###################################################

for file in files:
    path = dir+file
    t1 = time.time()
    frame = cv2.imread(path)
    frame = imutils.resize(frame, width=300, height=300)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)
    t2 = time.time()
    # print("[INFO] Time taken in reading image is {}".format(t2-t1))
    loading_image += (t2-t1)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    t1 = time.time()
    detections = net.forward()
    t2 = time.time()
    # print("[INFO] Total Inference time is {}".format(t2 - t1))
    inference_image += (t2-t1)
    # print (detections)
    # loop over the detections

    t1 = time.time()
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # print (startX,startY,endX,endY,)

            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    # show the output frame
    t2 = time.time()
    # print("[INFO] Total taken in visualisation is {}".format(t2 - t1))
    vis_image += (t2-t1)


print("\n\n-----------------------------------------")
print("Total images {}".format(total_images))
print("\n\n-----------------------------------------")
print("Total time in Reading {}".format(loading_image))
print("Total time in Inference {}".format(inference_image))
print("Total time in Visualisation {}".format(vis_image))

print("\n\n-----------------------------------------")
print("Average time in Reading {}".format(loading_image/total_images))
print("Average time in Inference {}".format(inference_image/total_images))
print("Average time in Visualisation {}".format(vis_image/total_images))



# cv2.imshow("Frame", frame)
# key = cv2.waitKey(1) & 0xFF
# k = cv2.waitKey(300000000)
# update the FPS counter
# fps.update()
 
# stop the timer and display FPS information
# fps.stop()
# print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
# cv2.destroyAllWindows()
# vs.stop()


