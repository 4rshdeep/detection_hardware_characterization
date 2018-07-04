# object detection hardware characterization


* run from root directory `python map_scripts/tf_object_detection.py`
* export model using `python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ./tensorflow_model/ssd_mobilenet_v1_coco_2017_11_17/ssd_mobilenet_v1_coco.config \
    --trained_checkpoint_prefix ./tensorflow_model/ssd_mobilenet_v1_coco_2017_11_17/model.ckpt \
    --output_directory ./tf_model_1.8`
    
