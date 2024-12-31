import cv2
import numpy as np
import os

def preprocess_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # 转换为 HSV 颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义蓝色的颜色范围
    lower_blue = np.array([100, 50, 50])  # 调整这些值根据实际需要
    upper_blue = np.array([145, 255, 255])

    # 创建颜色掩膜
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # 使用形态学操作清理掩膜
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return image, mask


def locate_license_plate(image, mask):
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    potential_plates = []
    for contour in contours:
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)

        # 筛选符合车牌比例的矩形区域
        aspect_ratio = w / h
        if 2 < aspect_ratio < 5:  # 车牌的长宽比通常在这个范围内
            potential_plates.append((x, y, w, h))

    # 在原图上绘制检测到的区域（可选）
    for (x, y, w, h) in potential_plates:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image, potential_plates


# 示例用法
if __name__ == "__main__":

    cur_path = os.getcwd()

    image_path = r"./CCPD2020/ccpd_green/test/0348036398467433-90_263-202&455_540&549-540&549_211&539_202&457_539&455-0_0_3_25_29_29_30_30-130-393.jpg"

    try:
        image, mask = preprocess_image(image_path)
        result_image, plates = locate_license_plate(image, mask)

        # 显示结果
        cv2.imshow("Mask", mask)
        cv2.imshow("Detected Plates", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except FileNotFoundError as e:
        print(e)