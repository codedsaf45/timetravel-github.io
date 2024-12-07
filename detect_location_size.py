from ultralytics import YOLO
import cv2

# YOLOv8 모델 로드 (pretrained 모델 사용)
model = YOLO('yolov8n.pt')  # YOLOv8 Nano 모델 사용 (적절히 변경 가능)

# 테스트 이미지 읽기
img = cv2.imread('test.jpg')  # 이미지 경로 변경 가능

# YOLOv8 모델 추론
results = model(img)

# 추론 결과에서 바운딩 박스 좌표와 크기 추출
for result in results:
    boxes = result.boxes  # 추론 결과의 바운딩 박스
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # 바운딩 박스 좌표 (좌상단, 우하단)
        width = x2 - x1  # 너비
        height = y2 - y1  # 높이
        conf = box.conf[0]  # confidence
        cls = int(box.cls[0])  # 클래스 번호

        # 바운딩 박스 정보 출력
        print(f"Class: {cls} | Confidence: {conf:.2f}")
        print(f"Bounding Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        print(f"Width: {width}, Height: {height}")

        # 바운딩 박스를 이미지 위에 그리기
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"{cls}:{conf:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 결과 이미지 보기
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
