# encoding:utf-8
import cv2
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import csv
import os

# fontC = ImageFont.truetype("Font/platech.ttf", 20, 0)

# Show image
def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def drawRectBox(image, rect, addText, fontC, color=(0,0,255)):
    """
    :param image: Original image
    :param rect: Rectangle coordinates
    :param addText: Class name
    :param fontC: Font
    :return:
    """
    # Draw position rectangle
    cv2.rectangle(image, (rect[0], rect[1]),
                 (rect[2], rect[3]),
                 color, 2)

    # Show Chinese
    # Adaptive font size
    font_size = int((rect[3]-rect[1])/1.5)
    fontC = ImageFont.truetype("Font/platech.ttf", font_size, 0)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)

    # Adjust text position (move text up by offset)
    offset = int(font_size * 1.5)  # Adjust this factor to control the upward movement
    text_position = (rect[0] + 2, rect[1] - offset)

    # Draw text on image
    draw.text(text_position, addText, (0, 0, 255), font=fontC)
    imagex = np.array(img)
    return imagex


def img_cvread(path):
    # Read file
    # img = cv2.imread(path)
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def draw_boxes(img, boxes):
    for each in boxes:
        x1 = each[0]
        y1 = each[1]
        x2 = each[2]
        y2 = each[3]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img


def cvimg_to_qpiximg(cvimg):
    height, width, depth = cvimg.shape
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    qimg = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)
    qpix_img = QPixmap(qimg)
    return qpix_img



