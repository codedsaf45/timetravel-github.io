import datetime
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import Point

CONFIDENCE_THRESHOLD = 0.6
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

model = YOLO('best (4).pt')
tracker = DeepSort(max_age=50)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

class DetectionPublisher(Node):
    def __init__(self):
        super().__init__('detection_publisher')
        # 사람 감지 여부와 위치를 발행할 퍼블리셔 설정
        self.detection_pub = self.create_publisher(Bool, 'person_detected', 10)
        self.position_pub = self.create_publisher(Point, 'person_position', 10)

    def publish_detection(self, detected, position=None):
        # 감지 여부 발행
        detection_msg = Bool()
        detection_msg.data = detected
        self.detection_pub.publish(detection_msg)
        
        # 위치 정보가 있을 경우 위치 발행
        if position:
            position_msg = Point()
            position_msg.x = position[0]
            position_msg.y = position[1]
            position_msg.z = 0.0  # Z축은 2D이므로 0으로 설정
            self.position_pub.publish(position_msg)
            self.get_logger().info(f'Published detection: {detected}, Position: {position}')

def main(args=None):
    rclpy.init(args=args)
    node = DetectionPublisher()

    try:
        while rclpy.ok():
            start = datetime.datetime.now()
            ret, frame = cap.read()
            if not ret:
                print('Cam Error')
                break

            detection = model.predict(source=[frame], save=False)[0]
            results = []

            for data in detection.boxes.data.tolist():
                confidence = float(data[4])
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                label = int(data[5])
                results.append([[xmin, ymin, xmax-xmin, ymax-ymin], confidence, label])

            tracks = tracker.update_tracks(results, frame=frame)

            person_detected = False
            person_position = None

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()

                # ID가 1번인 사람 감지 시 위치 설정
                if track_id == 1:
                    person_detected = True
                    xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
                    person_position = ((xmin + xmax) // 2, (ymin + ymax) // 2)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
                    cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
                    cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

            # 감지 결과 발행
            node.publish_detection(person_detected, person_position)

            end = datetime.datetime.now()
            total = (end - start).total_seconds()
            fps = f'FPS: {1 / total:.2f}'
            cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
