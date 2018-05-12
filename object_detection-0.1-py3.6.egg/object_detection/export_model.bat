@echo off

python export_inference_graph.py --input_type image_tensor --pipeline_config_path C:/Users/LOLiCON/Desktop/obj_detect/faster_rcnn_resnet101_car_pt_x_only_1.config --trained_checkpoint_prefix D:/DataSet/model_car/model.ckpt-22985 --output_directory D:/DataSet/model_car/fine_tuned_model
rem python Make_freeze_graph.py --model_folder C:/Anaconda3/Lib/site-packages/tensorflow/models/research/object_detection/fine_tuned_model --input_checkpoint C:/Anaconda3/Lib/site-packages/tensorflow/models/research/object_detection/models/train_aa/model.ckpt-1000
rem python export_inference_graph.py --input_type image_tensor --pipeline_config_path C:\Users\LOLiCON\Desktop\obj_detect\ssd_mobilenet_v1_car_pt_x.config --trained_checkpoint_prefix D:\DataSet\model_ssd\fine_tuned_model_ssd\model.ckpt --output_directory D:\DataSet\model_ssd\fine_tuned_model_ssd
rem python export_inference_graph.py --input_type image_tensor --pipeline_config_path C:/Users/LOLiCON/Desktop/obj_detect/faster_rcnn_resnet101_car_pt_x.config --trained_checkpoint_prefix D:/DataSet/model_car/model.ckpt-300 --output_directory D:/DataSet/model_car/fine_tuned_model
