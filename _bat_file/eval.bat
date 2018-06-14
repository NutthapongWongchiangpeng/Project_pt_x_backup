@echo off
set tensorflow_object_detection_api_path=C:\Users\ComSciSWU\Anaconda3\Lib\site-packages\tensorflow\models\research\object_detection
cd %tensorflow_object_detection_api_path%

set eval_dir=D:\Dataset\model_car_SSD_test\
python eval.py ^
--logtostderr ^
--checkpoint_dir=%eval_dir% ^
--eval_dir=%eval_dir% ^
--pipeline_config_path=%eval_dir%pipeline.config