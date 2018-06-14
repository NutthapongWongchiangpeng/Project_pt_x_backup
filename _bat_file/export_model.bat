@echo off
set tensorflow_object_detection_api_path=C:\Users\ComSciSWU\Anaconda3\Lib\site-packages\tensorflow\models\research\object_detection
cd %tensorflow_object_detection_api_path%

set input_type=image_tensor
set trained_model_path=D:\Dataset\model_car_SSD_test\
set pipeline_config_path=%trained_model_path%pipeline.config
set trained_step=41610
set checkpoint=%trained_model_path%model.ckpt-%trained_step%
set output_directory=%trained_model_path%fine_tuned_model

python export_inference_graph.py ^
--input_type %input_type% ^
--pipeline_config_path %pipeline_config_path%  ^
--trained_checkpoint_prefix %checkpoint% ^
--output_directory %output_directory%
