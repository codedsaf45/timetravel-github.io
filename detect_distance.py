import cv2
from ultralytics import YOLO

# YOLOv8 모델 로드 (pretrained 모델 또는 사용자 정의 모델 경로)
model = YOLO("best (4).pt")  # 기본적으로 YOLOv8n 사용, 필요에 따라 변경 가능

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 의미, 다른 카메라는 인덱스를 변경

# 웹캠에서 프레임을 읽어서 YOLOv8으로 감지
while True:
    ret, frame = cap.read()  # 프레임 읽기
    if not ret:
        print("웹캠에서 비디오를 가져올 수 없습니다.")
        break

    # 프레임 해상도 조정 (예: 640x480)
    frame = cv2.resize(frame, (640, 480))

    # YOLOv8로 감지 수행
    results = model(frame)

    # 감지된 객체를 루프를 통해 가져오기
    detections = results.pred[0]
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection  # 바운딩 박스 좌표 및 클래스 ID

        # 바운딩 박스와 클래스 정보 그리기
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"Class {int(class_id)}: {confidence:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과를 원본 프레임에 그리기
    annotated_frame = results[0].plot()

    # 결과 출력 (웹캠 창에 표시)
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
