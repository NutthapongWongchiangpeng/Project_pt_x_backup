python C:\Users\ComSciSWU\Anaconda3\Lib\site-packages\tensorflow/python/tools/optimize_for_inference.py^
  --input D:\Dataset\model_car_SSD_test\fine_tuned_model\frozen_inference_graph.pb^
  --output D:\Dataset\model_car_SSD_test\fine_tuned_model\opt_graph.pb^
  --input_names image_tensor^
  --output_names "num_detections,detection_scores,detection_boxes,detection_classes"^
  --placeholder_type_enum 4^
  --frozen_graph