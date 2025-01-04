#coding:utf-8
from ultralytics import YOLO
import cv2
import DetectTools as tools
from PIL import ImageFont
from paddleocr import PaddleOCR


def get_license_result(ocr,image):
    """
    image:input plate image
    output: license and confidence
    """
    result = ocr.ocr(image, cls=True)[0]
    if result:
        license_name, conf = result[0][1]
        if '·' in license_name:
            license_name = license_name.replace('·', '')
        return license_name, conf
    else:
        return None, None

# Test Image Path
img_path = "TestFiles/015859375-90_261-217&496_449&565-449&565_226&559_217&496_448&497-0_0_3_24_33_33_33_25-99-69.jpg"
now_img = tools.img_cvread(img_path)

fontC = ImageFont.truetype("Font/platech.ttf", 50, 0)
# Paddleocr model
cls_model_dir='paddleModels/whl/cls/ch_ppocr_mobile_v2.0_cls_infer'
rec_model_dir='paddleModels/whl/rec/ch/ch_PP-OCRv4_rec_infer'
ocr = PaddleOCR(use_angle_cls=False, lang="ch", det=False, cls_model_dir=cls_model_dir,rec_model_dir=rec_model_dir)

# Yolo trained model
path = 'models/best.pt'
# Load model
# conf	0.25	object confidence threshold for detection
# iou	0.7	int.ersection over union (IoU) threshold for NMS
model = YOLO(path, task='detect')
# model = YOLO(path, task='detect',conf=0.5)
# Detect Image
results = model(img_path)[0]

location_list = results.boxes.xyxy.tolist()
if len(location_list) >= 1:
    location_list = [list(map(int, e)) for e in location_list]
    # Get plate area picture
    license_imgs = []
    for each in location_list:
        x1, y1, x2, y2 = each
        cropImg = now_img[y1:y2, x1:x2]
        license_imgs.append(cropImg)
        cv2.imshow('PlateImage',cropImg)
        cv2.waitKey(0)
    # License Detection Result
    lisence_res = []
    conf_list = []
    for each in license_imgs:
        license_num, conf = get_license_result(ocr, each)
        if license_num:
            lisence_res.append(license_num)
            conf_list.append(conf)
        else:
            lisence_res.append('Can not detect')
            conf_list.append(0)
    print(f"Detected plate: {lisence_res[0]}, Confidence: {conf_list[0]}")
    for text, box in zip(lisence_res, location_list):
        now_img = tools.drawRectBox(now_img, box, text, fontC)

now_img = cv2.resize(now_img,dsize=None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
cv2.imshow("YOLOv8 Detection", now_img)
cv2.waitKey(0)
