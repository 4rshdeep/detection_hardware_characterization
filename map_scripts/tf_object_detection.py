# coding: utf-8

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

CONFIDENCE=0.4
DPI=96
cwd = os.getcwd()

# What model to download.
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'tensorflow-model/'+ MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('inference_scripts/data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = '/home/arshdeep/mAP/images'
tmpImgs = os.listdir(PATH_TO_TEST_IMAGES_DIR)
TEST_IMAGE_PATHS = []
for img in tmpImgs:
  TEST_IMAGE_PATHS.append(os.path.join(PATH_TO_TEST_IMAGES_DIR, img))



with detection_graph.as_default():
  with tf.Session() as sess:
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes'
    ]:
      tensor_name = key + ':0'
      if tensor_name in all_tensor_names:
        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

    for j,image_path in enumerate(TEST_IMAGE_PATHS):
      image = Image.open(image_path)
      width, height = image.size
      image_np = load_image_into_numpy_array(image)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image_np, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      # if 'detection_masks' in output_dict:
        # output_dict['detection_masks'] = output_dict['detection_masks'][0]
      f = open(cwd+"/predicted/"+tmpImgs[j][0:tmpImgs[j].index('.')]+".txt",'w')

      for i,score in enumerate(output_dict['detection_scores']):
        if score > CONFIDENCE and category_index[int(output_dict['detection_classes'][i])]['name']!='dontcare':
          org_detection_box = list(output_dict['detection_boxes'][i])
          output_dict['detection_boxes'][i][0]=org_detection_box[1]*width
          output_dict['detection_boxes'][i][1]=org_detection_box[0]*height
          output_dict['detection_boxes'][i][2]=org_detection_box[3]*width
          output_dict['detection_boxes'][i][3]=org_detection_box[2]*height
          label = category_index[int(output_dict['detection_classes'][i])]['name']
          startX = output_dict['detection_boxes'][i][0]
          startY = output_dict['detection_boxes'][i][1]
          endX = output_dict['detection_boxes'][i][2]
          endY = output_dict['detection_boxes'][i][3]
          f.write(str(label) + " " + str(score) + " " + str(startX)  + " " + str(startY) + " " + str(endX)+ " " + str(endY)+"\n")
      f.close()
