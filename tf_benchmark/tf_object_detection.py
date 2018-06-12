# coding: utf-8
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import time 
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
# # import matplotlib
# matplotlib.use('TkAgg')
import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("..")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

CONFIDENCE=0.35
DPI=96

# What model to download.
# PATH_TO_FROZEN_GRAPH = "../tensorflow_model/ssd_mobilenet_v1.pb"
PATH_TO_FROZEN_GRAPH = "../tensorflow_model/ssd_mobilenet_v1.pb"
PATH_TO_LABELS = os.path.join('../tensorflow_model/data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90


# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# ## Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# PATH_TO_TEST_IMAGES_DIR = 'test_images'
PATH_TO_TEST_IMAGES_DIR = '../benchmark/images/'

tmpImgs = os.listdir(PATH_TO_TEST_IMAGES_DIR)
TEST_IMAGES = []

for img in tmpImgs:
  TEST_IMAGES.append(os.path.join(PATH_TO_TEST_IMAGES_DIR, img))

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


with detection_graph.as_default():
  with tf.Session() as sess:
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
      tensor_name = key + ':0'
      if tensor_name in all_tensor_names:
        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

    ########## First pass empty ####################
    img = TEST_IMAGES[0]
    img = Image.open(img)
    img = load_image_into_numpy_array(img)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    t1 = time.time()
    sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(img, 0)})
    t2 = time.time()
    fst_inference = (t2-t1)


    inf_time  = 0
    read_time = 0
    vis_time  = 0

    for image_path in TEST_IMAGES:
      
      t1 = time.time()
      image = Image.open(image_path)
      width, height = image.size
      image = load_image_into_numpy_array(image)
      t2 = time.time()
      print("[INFO] Time taken in loading image is {}".format(t2-t1))
      read_time += (t2-t1)

      # Run inference
      t1 = time.time()
      output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
      t2 = time.time()
      print("[INFO] Time taken in inference is {}".format(t2-t1))
      inf_time += (t2-t1)

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        print("detection_masks ftw!")
        output_dict['detection_masks'] = output_dict['detection_masks'][0]

      t1 = time.time()
      vis_util.visualize_boxes_and_labels_on_image_array(
              image,
              output_dict['detection_boxes'],
              output_dict['detection_classes'],
              output_dict['detection_scores'],
              category_index,
              instance_masks=output_dict.get('detection_masks'),
              use_normalized_coordinates=True,
              line_thickness=5)
      t2 = time.time()
      print("[INFO] Time taken in Visualization is {}".format(t2-t1))
      print ("===========================")


num_imgs = len(TEST_IMAGES)

print("\n[INFO] Time taken in First inference is {}".format(fst_inference))

print("\n[INFO] Total Time taken in loading image is {}".format(read_time))    
print("[INFO] Total Time taken in inference is {}".format(inf_time))
print("[INFO] Total Time taken in Visualization is {}".format(vis_time))


print("\n[INFO] Average Time taken in loading image is {}".format(read_time/num_imgs))    
print("[INFO] Average Time taken in inference is {}".format(inf_time/num_imgs))
print("[INFO] Average Time taken in Visualization is {}".format(vis_time/num_imgs))

      

