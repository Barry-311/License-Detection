# coding:utf-8
from ultralytics import YOLO
import cv2
# model address
path = 'models/best.pt'
# picture address
img_path = "TestFiles/013828125-90_259-275&449_491&514-491&514_285&511_275&450_485&449-0_0_3_25_33_33_31_33-164-249.jpg"


# Load the trained model
# conf	0.25	object confidence threshold for detection
# iou	0.7	intersection over union (IoU) threshold for NMS
model = YOLO(path, task='detect')
# model = YOLO(path, task='detect',conf=0.5)


# Detect image
results = model(img_path)
res = results[0].plot()
# res = cv2.resize(res,dsize=None,fx=0.3,fy=0.3,interpolation=cv2.INTER_LINEAR)
cv2.imshow("YOLOv8 Detection", res)
cv2.waitKey(0)
