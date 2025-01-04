#coding:utf-8
from ultralytics import YOLO
import cv2
import DetectTools as tools
from PIL import ImageFont, ImageDraw, Image
from paddleocr import PaddleOCR
import numpy as np
import time

def get_license_result(ocr, image):
    """
    image: input plate image
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

# Initialize OCR model
cls_model_dir = 'paddleModels/whl/cls/ch_ppocr_mobile_v2.0_cls_infer'
rec_model_dir = 'paddleModels/whl/rec/ch/ch_PP-OCRv4_rec_infer'
ocr = PaddleOCR(use_angle_cls=False, lang="ch", det=False, cls_model_dir=cls_model_dir, rec_model_dir=rec_model_dir)

# Initialize Yolo
path = 'models/best.pt'
model = YOLO(path)

# load font
fontC = ImageFont.truetype("Font/platech.ttf", 30, 0)  # 调整字体大小

# Use video
video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

# get frame
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_duration = 1 / fps  # target time of each frame
max_frame_duration = frame_duration * 2

# Initialize
current_overlay = None
current_license = None  # save current number

# Use preprocessing or not
enable_preprocessing = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # YOLO detection
    results = model(frame, conf=0.25)
    location_list = []

    if results and results[0].boxes.xyxy is not None:
        # get bounding box
        location_list = results[0].boxes.xyxy.cpu().numpy().tolist()

    if len(location_list) >= 1:
        location_list = [list(map(int, e[:4])) for e in location_list]
        license_imgs = []

        # get plate area
        for each in location_list:
            x1, y1, x2, y2 = each
            crop_img = frame[y1:y2, x1:x2]

            # preprocessing
            if enable_preprocessing:
                gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                kernel = np.array([[-1, -1, -1],
                                   [-1,  8, -1],
                                   [-1, -1, -1]])
                preprocessed_img = cv2.filter2D(gray_img, -1, kernel)
                license_imgs.append(preprocessed_img)
            else:
                license_imgs.append(crop_img)

        # PaddleOCR
        lisence_res = []
        conf_list = []
        for each in license_imgs:
            if len(each.shape) == 2:
                each = cv2.cvtColor(each, cv2.COLOR_GRAY2BGR)
            license_num, conf = get_license_result(ocr, each)
            if license_num:
                lisence_res.append(license_num)
                conf_list.append(conf)
            else:
                lisence_res.append('can not recognize')
                conf_list.append(0)

        # draw result
        for text, box, conf, img in zip(lisence_res, location_list, conf_list, license_imgs):
            frame = tools.drawRectBox(frame, box, text, fontC)

            # update license info
            if conf > 0.95 and text != current_license:
                current_license = text
                overlay_height = frame.shape[0] // 4
                overlay_width = frame.shape[1] // 6
                overlay = np.zeros((overlay_height, overlay_width, 3), dtype=np.uint8)

                # draw plate image
                plate_resized = cv2.resize(img, (overlay_width, overlay_height // 2))
                if len(plate_resized.shape) == 2:  # grey - 2 dimension
                    plate_resized = cv2.cvtColor(plate_resized, cv2.COLOR_GRAY2BGR)
                overlay[:overlay_height // 2, :, :] = plate_resized

                # plate info & con
                pil_overlay = Image.fromarray(overlay)
                draw = ImageDraw.Draw(pil_overlay)
                draw.text((10, overlay_height // 2 + 10), text, font=fontC, fill=(255, 255, 255))  # plate info
                draw.text((10, overlay_height // 2 + 50), f"Conf: {conf:.2f}", font=fontC,
                          fill=(255, 255, 255))  # confidence
                current_overlay = np.array(pil_overlay)  # update

    if current_overlay is not None:
        overlay_height, overlay_width = current_overlay.shape[:2]
        frame[:overlay_height, -overlay_width:, :] = current_overlay

    # show video processing
    cv2.imshow("YOLOv8 License Plate Detection", frame)

    # calculate time
    elapsed_time = time.time() - start_time
    if elapsed_time > max_frame_duration:
        # skip frames
        skip_frames = int(elapsed_time / frame_duration) - 1
        for _ in range(skip_frames):
            cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press Q to exit
        break

# release resource
cap.release()
cv2.destroyAllWindows()
