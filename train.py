#coding:utf-8
from ultralytics import YOLO
import os
import yaml


def update_paths_in_yaml():
    # 假设 train.py 所在文件夹为项目根目录
    project_root = os.getcwd()
    yaml_path = project_root + "/datasets/PlateData/data.yaml"


    # Check if the yaml file exists
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"{yaml_path} not found.")
    # Load the YAML content
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    data['train']=project_root + "/datasets/PlateData/images/train"
    data['val'] = project_root + "/datasets/PlateData/images/val"
    data['test'] = project_root + "/datasets/PlateData/images/test"

    # Save the updated YAML content back to the file
    with open(yaml_path, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)


# Load yolo model
model = YOLO("yolov8n.pt")
# Use the model
if __name__ == '__main__':
    update_paths_in_yaml()
    # Use the model
    results = model.train(data='datasets/PlateData/data.yaml', epochs=300, batch=4)  # 训练模型
    # transfer to onnx
    # success = model.export(format='onnx')

