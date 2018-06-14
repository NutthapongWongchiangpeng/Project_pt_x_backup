# This script is used to demonstrate MobileNet-SSD network using OpenCV deep learning module.
#
# It works with model taken from https://github.com/chuanqi305/MobileNet-SSD/ that
# was trained in Caffe-SSD framework, https://github.com/weiliu89/caffe/tree/ssd.
# Model detects objects from 20 classes.
#
# Also TensorFlow model from TensorFlow object detection model zoo may be used to
# detect objects from 90 classes:
# http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
# Text graph definition must be taken from opencv_extra:
# https://github.com/opencv/opencv_extra/tree/master/testdata/dnn/ssd_mobilenet_v1_coco.pbtxt
import numpy as np
import argparse

try:
    import cv2 as cv
except ImportError:
    raise ImportError('Can\'t find OpenCV Python module. If you\'ve built it from sources without installation, '
                      'configure environment variable PYTHONPATH to "opencv_build_dir/lib" directory (with "python3" subdirectory if required)')

inWidth = 300
inHeight = 300
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to run MobileNet-SSD object detection network '
                    'trained either in Caffe or TensorFlow frameworks.')
    parser.add_argument("--video",default="output2.mp4", help="path to video file. If empty, camera's stream will be used")
    parser.add_argument("--prototxt", default="D:/Dataset/model_car_SSD_test/fine_tuned_model/frozen_inference_graph.pbtxt",
                                      help='Path to text network file: '
                                           'MobileNetSSD_deploy.prototxt for Caffe model or '
                                           'ssd_mobilenet_v1_coco.pbtxt from opencv_extra for TensorFlow model')
    parser.add_argument("--weights", default="D:/Dataset/model_car_SSD_test/fine_tuned_model/frozen_inference_graph.pb",
                                     help='Path to weights: '
                                          'MobileNetSSD_deploy.caffemodel for Caffe model or '
                                          'frozen_inference_graph.pb from TensorFlow.')
    parser.add_argument("--num_classes", default=4, type=int,
                        help="Number of classes. It's 20 for Caffe model from "
                             "https://github.com/chuanqi305/MobileNet-SSD/ and 90 for "
                             "TensorFlow model from http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz")
    parser.add_argument("--thr", default=0.5, type=float, help="confidence threshold to filter out weak detections")
    args = parser.parse_args()


    net = cv.dnn.readNetFromTensorflow(args.weights, args.prototxt)
    swapRB = True
    classNames = { 0: 'background', 1: 'car', 2: 'bus', 3: 'van', 4: 'other' }

    if args.video:
        cap = cv.VideoCapture(args.video)
    else:
        cap = cv.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        blob = cv.dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), (meanVal, meanVal, meanVal), swapRB)
        net.setInput(blob)
        detections = net.forward()

        cols = frame.shape[1]
        rows = frame.shape[0]

        if cols / float(rows) > WHRatio:
            cropSize = (int(rows * WHRatio), rows)
        else:
            cropSize = (cols, int(cols / WHRatio))

        y1 = int((rows - cropSize[1]) / 2)
        y2 = y1 + cropSize[1]
        x1 = int((cols - cropSize[0]) / 2)
        x2 = x1 + cropSize[0]
        frame = frame[y1:y2, x1:x2]

        cols = frame.shape[1]
        rows = frame.shape[0]

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > args.thr:
                class_id = int(detections[0, 0, i, 1])
                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop   = int(detections[0, 0, i, 5] * cols)
                yRightTop   = int(detections[0, 0, i, 6] * rows)

                cv.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                              (0, 255, 0))
                if class_id in classNames:
                    con=('%.4f' % confidence).rstrip('0').rstrip('.')
                    label = classNames[class_id] + ": " + con#str(confidence)
                    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                         (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                         (255, 255, 255), cv.FILLED)
                    cv.putText(frame, label, (xLeftBottom, yLeftBottom),
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow("detections", frame)
        if cv.waitKey(1) >= 0:
            break
