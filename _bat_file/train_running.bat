@echo off
set tensorflow_object_detection_api_path=C:\Users\ComSciSWU\Anaconda3\Lib\site-packages\tensorflow\models\research\object_detection
cd %tensorflow_object_detection_api_path%

set train_dir=D:\DataSet\model_car_SSD_test
set pipeline_config_path=D:\Dataset\Obj_detec\ssd_mobilenet_v1_car.config

python train.py --logtostderr ^
--train_dir %train_dir% ^
--pipeline_config_path %pipeline_config_path%
