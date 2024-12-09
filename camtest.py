import cv2

# GStreamer 파이프라인
gstreamer_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM),width=3280,height=2464,format=(string)NV12,framerate=(fraction)20/1 ! "
    "nvvidconv ! "
    "video/x-raw,format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw,format=(string)BGR ! appsink"
)

# OpenCV VideoCapture로 GStreamer 파이프라인 열기
cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("CSI 카메라를 열 수 없습니다.")
    exit(-1)

# 비디오 프레임 읽기 및 표시
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    cv2.imshow("CSI Camera", frame)

    # 'q'를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
