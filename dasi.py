import cv2
import threading
import queue
from ultralytics import YOLO
from pyorbbecsdk import Config, OBSensorType, OBFormat, Pipeline, FrameSet, VideoStreamProfile, OBError
import sys
sys.path.append("pyorbbecsdk/examples")
from utils import frame_to_bgr_image

ESC_KEY = 27

# YOLO 모델 로드
try:
    model = YOLO("best (4).pt").to("cuda")
except Exception as e:
    print("CUDA를 사용할 수 없습니다. CPU로 전환합니다.")
    model = YOLO("best (4).pt")

# 원본 프레임과 탐지된 프레임을 저장할 큐 생성
frame_queue = queue.Queue(maxsize=10)
detected_frame_queue = queue.Queue(maxsize=10)

class YoloDetect(threading.Thread):
    def __init__(self, frame_queue, detected_frame_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.detected_frame_queue = detected_frame_queue
        self.running = True

    def run(self):
        global x_center, y_center
        while self.running:
            if not self.frame_queue.empty():
                # 큐에서 프레임을 가져옴
                frame = self.frame_queue.get()
                
                # YOLO로 탐지 수행
                results = model(frame)
                
                # 예시로 탐지 결과 그리기
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        x_center = (x1 + x2) // 2
                        y_center = (y1 + y2) // 2
                        label = f"x_center: {x_center}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 중심에 빨간색 점 표시
                        #cv2.circle(frame, (x_center, y_center), 5, (0, 0, 255), -1)
                        
                        
                        
                
                # 탐지된 프레임을 detected_frame_queue에 넣기
                if not self.detected_frame_queue.full():
                    self.detected_frame_queue.put(frame)

    def stop(self):
        self.running = False


def main():
    config = Config()
    pipeline = Pipeline()
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
        except OBError as e:
            print(e)
            color_profile = profile_list.get_default_video_stream_profile()
            print("color profile: ", color_profile)
        config.enable_stream(color_profile)
    except Exception as e:
        print(e)
        return

    # YOLO 탐지 스레드 시작
    yolo_thread = YoloDetect(frame_queue, detected_frame_queue)
    yolo_thread.start()

    pipeline.start(config)
    while True:
        try:
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            
            # 프레임을 BGR 이미지로 변환
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("failed to convert frame to image")
                continue
            
            # 프레임을 frame_queue에 추가하여 YOLO 스레드로 전달
            if not frame_queue.full():
                frame_queue.put(color_image)
            
            # 탐지 결과 프레임이 있는 경우 가져오기
            if not detected_frame_queue.empty():
                detected_frame = detected_frame_queue.get()
                # 탐지된 프레임 표시
                cv2.imshow("YOLO Detection", detected_frame)
                print(x_center, y_center)

            # 원본 영상 표시
            cv2.imshow("Color Viewer", color_image)
            print()
            
            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break
        except KeyboardInterrupt:
            break

    # 스레드 중지 및 종료
    yolo_thread.stop()
    yolo_thread.join()
    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
