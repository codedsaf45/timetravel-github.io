import cv2
import threading
from ultralytics import YOLO
from pyorbbecsdk import Config, OBError, OBSensorType, OBFormat, Pipeline, FrameSet, VideoStreamProfile
import sys
sys.path.append('pyorbbecsdk/examples')
from utils import frame_to_bgr_image

ESC_KEY = 27

# YOLO 모델 설정
try:
    model = YOLO("best (4).pt").to("cuda")
except Exception as e:
    print("CUDA를 사용할 수 없습니다. CPU로 전환합니다.")
    model = YOLO("best (4).pt")

# 프레임 캡처용 스레드 클래스
class FrameCaptureThread(threading.Thread):
    def __init__(self, pipeline, yolo_thread):
        super().__init__()
        self.pipeline = pipeline
        self.yolo_thread = yolo_thread
        self.running = True

    def run(self):
        while self.running:
            try:
                frames: FrameSet = self.pipeline.wait_for_frames(500)
                if frames:
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        frame = frame_to_bgr_image(color_frame)
                        self.yolo_thread.set_frame(frame)
            except OBError as e:
                print("Frame capture error:", e)

    def stop(self):
        self.running = False

# YOLO 객체 탐지용 스레드 클래스
class YOLODetectionThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.frame = None
        self.results = None
        self.running = True
        self.lock = threading.Lock()

    def set_frame(self, frame):
        with self.lock:
            self.frame = frame

    def run(self):
        while self.running:
            if self.frame is not None:
                with self.lock:
                    frame = self.frame
                    self.frame = None
                self.results = model(frame)

    def stop(self):
        self.running = False

def main():
    config = Config()
    pipeline = Pipeline()

    # Color sensor 설정
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
        config.enable_stream(color_profile)
    except Exception as e:
        print("스트림 설정 오류:", e)
        return

    pipeline.start(config)

    # YOLO 객체 탐지 스레드 시작
    yolo_thread = YOLODetectionThread()
    yolo_thread.start()

    # 프레임 캡처 스레드 시작
    capture_thread = FrameCaptureThread(pipeline, yolo_thread)
    capture_thread.start()

    try:
        while True:
            if yolo_thread.results is not None:
                frame = yolo_thread.frame
                results = yolo_thread.results

                # 탐지된 객체 표시
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = box.conf[0]
                        class_id = int(box.cls[0])

                        if class_id == 0:  # 사람 클래스
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{model.names[class_id]} {confidence:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # 결과 이미지 출력
                if frame is not None:
                    cv2.imshow("Object Detection", frame)
                    print("Displaying frame")  # 디버깅 로그 추가
                
                else:
                    print("Frame is None or empty")  # 디버깅 로그 추가
                

            # 키 입력 대기
            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break
    except KeyboardInterrupt:
        pass
    finally:
        # 스레드 종료 및 리소스 정리
        capture_thread.stop()
        capture_thread.join()
        yolo_thread.stop()
        yolo_thread.join()
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()