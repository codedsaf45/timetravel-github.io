import torch
import onnx
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO("best (4).pt")  # best.pt는 YOLOv8n 모델로 학습된 가중치 파일입니다.

# 모델을 ONNX로 변환
dummy_input = torch.randn(1, 3, 640, 640)  # YOLOv8의 입력 크기 (1, 3, 640, 640)
onnx_path = "best.onnx"

# 모델을 ONNX 형식으로 내보내기
model.export(format="onnx", opset=12, dynamic=True)  # opset 및 dynamic 설정은 필요에 따라 조정
