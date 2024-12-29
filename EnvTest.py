import paddle
from paddleocr import PaddleOCR
import shapely

print(paddle.__version__)  # 打印 PaddlePaddle 版本号
print(paddle.is_compiled_with_cuda())
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # 你可以更换为 'ch' 等其他语言
print(shapely.__version__)