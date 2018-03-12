@echo off
rem python train.py --logtostderr --train_dir=D:\DataSet\model_car --pipeline_config_path=C:\Users\LOLiCON\Desktop\obj_detect\faster_rcnn_resnet101_car_pt_x.config rem >> train_running.log
rem python train.py --logtostderr --train_dir=D:\DataSet\model --pipeline_config_path=C:\Users\LOLiCON\Desktop\obj_detect\faster_rcnn_resnet152_pets_pt_x.config
python train.py --logtostderr --train_dir=D:\DataSet\model_ssd --pipeline_config_path=C:\Users\LOLiCON\Desktop\obj_detect\ssd_mobilenet_v1_car_pt_x.config
