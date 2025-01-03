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
    image: 输入的车牌截取照片
    输出：车牌号与置信度
    """
    result = ocr.ocr(image, cls=True)[0]
    if result:
        license_name, conf = result[0][1]
        if '·' in license_name:
            license_name = license_name.replace('·', '')
        return license_name, conf
    else:
        return None, None

# 初始化PaddleOCR模型
cls_model_dir = 'paddleModels/whl/cls/ch_ppocr_mobile_v2.0_cls_infer'
rec_model_dir = 'paddleModels/whl/rec/ch/ch_PP-OCRv4_rec_infer'
ocr = PaddleOCR(use_angle_cls=False, lang="ch", det=False, cls_model_dir=cls_model_dir, rec_model_dir=rec_model_dir)

# 初始化YOLO模型
path = 'models/best.pt'
model = YOLO(path, task='detect')
print(model._version)

# 加载字体
fontC = ImageFont.truetype("Font/platech.ttf", 30, 0)  # 调整字体大小

# 处理视频
video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频帧率
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_duration = 1 / fps  # 每帧的目标时间
max_frame_duration = frame_duration * 2  # 超过此时间则跳帧

# 初始化图层状态变量
current_overlay = None
current_license = None  # 保存当前显示的车牌号

# 开关：是否启用预处理
enable_preprocessing = True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # YOLO检测
    results = model(frame)[0]
    location_list = results.boxes.xyxy.tolist()

    if len(location_list) >= 1:
        location_list = [list(map(int, e)) for e in location_list]
        license_imgs = []

        # 提取车牌区域
        for each in location_list:
            x1, y1, x2, y2 = each
            crop_img = frame[y1:y2, x1:x2]

            # 预处理：灰度化 + 高通滤波
            if enable_preprocessing:
                gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                kernel = np.array([[-1, -1, -1],
                                   [-1,  8, -1],
                                   [-1, -1, -1]])
                preprocessed_img = cv2.filter2D(gray_img, -1, kernel)
                license_imgs.append(preprocessed_img)
            else:
                license_imgs.append(crop_img)

        # PaddleOCR识别车牌号
        lisence_res = []
        conf_list = []
        for each in license_imgs:
            # OCR 输入需为三通道
            if len(each.shape) == 2:
                each = cv2.cvtColor(each, cv2.COLOR_GRAY2BGR)
            license_num, conf = get_license_result(ocr, each)
            if license_num:
                lisence_res.append(license_num)
                conf_list.append(conf)
            else:
                lisence_res.append('无法识别')
                conf_list.append(0)

        # 在视频帧上绘制检测结果
        for text, box, conf, img in zip(lisence_res, location_list, conf_list, license_imgs):
            frame = tools.drawRectBox(frame, box, text, fontC)

            # 如果置信度大于 0.95 且车牌内容发生变化，更新右上角图层
            if conf > 0.95 and text != current_license:
                current_license = text  # 更新当前车牌内容
                overlay_height = frame.shape[0] // 4
                overlay_width = frame.shape[1] // 6
                overlay = np.zeros((overlay_height, overlay_width, 3), dtype=np.uint8)

                # 将车牌图片放置在图层中
                plate_resized = cv2.resize(img, (overlay_width, overlay_height // 2))
                # 如果当前图像是灰度图，则转换为三通道
                if len(plate_resized.shape) == 2:  # 灰度图只有 2 个维度
                    plate_resized = cv2.cvtColor(plate_resized, cv2.COLOR_GRAY2BGR)
                overlay[:overlay_height // 2, :, :] = plate_resized

                # 在图层上绘制车牌信息和置信度（两行）
                pil_overlay = Image.fromarray(overlay)
                draw = ImageDraw.Draw(pil_overlay)
                draw.text((10, overlay_height // 2 + 10), text, font=fontC, fill=(255, 255, 255))  # 第一行：车牌信息
                draw.text((10, overlay_height // 2 + 50), f"Conf: {conf:.2f}", font=fontC,
                          fill=(255, 255, 255))  # 第二行：置信度
                current_overlay = np.array(pil_overlay)  # 更新当前图层

    # 如果有激活的图层，将其显示在右上角
    if current_overlay is not None:
        overlay_height, overlay_width = current_overlay.shape[:2]
        frame[:overlay_height, -overlay_width:, :] = current_overlay

    # 显示实时处理视频
    cv2.imshow("YOLOv8 License Plate Detection", frame)

    # 计算帧处理时间
    elapsed_time = time.time() - start_time
    if elapsed_time > max_frame_duration:
        # 跳过若干帧
        skip_frames = int(elapsed_time / frame_duration) - 1
        for _ in range(skip_frames):
            cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
