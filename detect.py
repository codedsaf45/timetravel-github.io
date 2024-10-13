import cv2
from ultralytics import YOLO

# YOLOv8 모델 로드 (pretrained 모델 또는 사용자 정의 모델 경로)
model = YOLO("best.pt")  # 기본적으로 YOLOv8n 사용, 필요에 따라 변경 가능

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 의미, 다른 카메라는 인덱스를 변경

# 웹캠에서 프레임을 읽어서 YOLOv8으로 감지
while True:
    ret, frame = cap.read()  # 프레임 읽기
    if not ret:
        print("웹캠에서 비디오를 가져올 수 없습니다.")
        break

    # YOLOv8로 감지 수행
    results = model(frame)

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
