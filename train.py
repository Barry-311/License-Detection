#coding:utf-8
from ultralytics import YOLO

# Load yolo model
model = YOLO("yolov8n.pt")
# Use the model
if __name__ == '__main__':
    # Use the model
    results = model.train(data='datasets/PlateData/data.yaml', epochs=300, batch=4)  # 训练模型
    # transfer to onnx
    # success = model.export(format='onnx')



