import cv2 as cv

MODEL_NAME = 'D:/Dataset/model_car_SSD_test/fine_tuned_model/'
PATH_TO_PB = MODEL_NAME + 'frozen_inference_graph.pb'

PATH_TO_PBTXT = MODEL_NAME + 'frozen_inference_graph.pbtxt'

cvNet = cv.dnn.readNetFromTensorflow(PATH_TO_PB, PATH_TO_PBTXT)

img = cv.imread('D:/DataSet/Insight-MVT_Annotation_Train/MVI_20051/img00130.jpg')
rows = img.shape[0]
cols = img.shape[1]
cvNet.setInput(cv.dnn.blobFromImage(img, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
cvOut = cvNet.forward()

for detection in cvOut[0,0,:,:]:
    score = float(detection[2])
    if score > 0.3:
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

cv.imshow('img', img)
cv.waitKey()