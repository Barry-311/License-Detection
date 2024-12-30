import paddle
from paddleocr import PaddleOCR
import shapely

print(paddle.__version__)  # print PaddlePaddle version
print(paddle.is_compiled_with_cuda())
ocr = PaddleOCR(use_angle_cls=True, lang='en')
print(shapely.__version__)