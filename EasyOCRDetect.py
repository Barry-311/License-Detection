#coding:utf-8
from ultralytics import YOLO
import cv2
import DetectTools as tools
from PIL import ImageFont, ImageDraw, Image
from easyocr import Reader
import numpy as np
import time

def get_license_result(ocr, image):
    """
    image: 输入的车牌截取照片
    输出：车牌号与置信度
    """
    result = ocr.readtext(image)
    if result:
        license_name, conf = result[0][1], result[0][2]
        if '·' in license_name:
            license_name = license_name.replace('·', '')
        return license_name, conf
    else:
        return None, None

# 初始化EasyOCR模型
ocr = Reader(['ch_sim', 'en'])

# 初始化YOLO模型
path = 'models/best.pt'
model = YOLO(path, task='detect')

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
enable_preprocessing = False

# 播放控制开关
paused = False

while cap.isOpened():
    if not paused:
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

                # 预处理：灰度化 + 自动阈值二值化 + 开运算
                if enable_preprocessing:
                    gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
                    binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY, 11, 2)  # 自适应阈值
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    preprocessed_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)  # 开运算

                    # 转换为三通道以兼容 EasyOCR
                    preprocessed_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2BGR)
                    license_imgs.append(preprocessed_img)
                else:
                    license_imgs.append(crop_img)

            # EasyOCR识别车牌号
            lisence_res = []
            conf_list = []
            for each in license_imgs:
                license_num, conf = get_license_result(ocr, each)
                if license_num:
                    lisence_res.append(license_num)
                    conf_list.append(conf)
                else:
                    lisence_res.append('OCR Failed')
                    conf_list.append(0)

            # 在视频帧上绘制检测结果
            for text, box, conf, img in zip(lisence_res, location_list, conf_list, license_imgs):
                frame = tools.drawRectBox(frame, box, text, fontC)

                # 如果置信度大于 0.95 且车牌内容发生变化，更新右上角图层
                if conf > 0.75 and text != current_license:
                    current_license = text  # 更新当前车牌内容
                    overlay_height = frame.shape[0] // 4
                    overlay_width = frame.shape[1] // 6
                    overlay = np.zeros((overlay_height, overlay_width, 3), dtype=np.uint8)

                    # 将车牌图片放置在图层中
                    plate_resized = cv2.resize(img, (overlay_width, overlay_height // 2))
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
            print("The duration of this frame is ", elapsed_time, "\n")
            skip_frames = int(elapsed_time / frame_duration) - 1
            for _ in range(skip_frames):
                cap.read()

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 按 'q' 键退出
        break
    elif key == ord(' '):  # 按空格键暂停或恢复
        paused = not paused

# 释放资源
cap.release()
cv2.destroyAllWindows()
