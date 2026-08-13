import cv2
import time
import numpy as np
import mss
try:
    from rpycrsf import Drone
except ImportError:
    Drone = None
from ultralytics import YOLO

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
TEST_MODE = True           
USE_SCREEN_CAPTURE = False 

CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30
CENTER_X = FRAME_WIDTH // 2
CENTER_Y = FRAME_HEIGHT // 2

TARGET_TRASH_AREA = 25000  
TARGET_BIN_AREA = 15000    

YAW_KP = 0.0015
PITCH_KP = 0.00005
THROTTLE_KP = 0.002
BASE_THROTTLE = 0.5 

# 고급 비행 제어 설정 (Jitter 방지 및 유예 시간)
DEADZONE_X = 30     # 좌우 데드존 (픽셀)
DEADZONE_Y = 20     # 상하 데드존 (픽셀)
GRACE_PERIOD = 1.5  # 목표물 분실 시 유예 시간 (초)

# ==========================================
# 2. 드론 상태 (State Machine)
# ==========================================
STATE_SEARCHING_TRASH = "Search: Trash"
STATE_TRACKING_TRASH  = "Track: Trash"
STATE_PICKING_UP      = "Action: Pickup"
STATE_SEARCHING_BIN   = "Search: Bin"
STATE_TRACKING_BIN    = "Track: Bin"
STATE_DROPPING_OFF    = "Action: Dropoff"
STATE_DONE            = "Mission Complete"

# ==========================================
# 3. 비전 인공지능 (YOLO & ArUco) 초기화
# ==========================================
model = YOLO('best.pt')
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
TARGET_BIN_ID = 0

# ==========================================
# 4. 기능 함수 (Functions)
# ==========================================
def get_yolo_target(frame):
    """ YOLO 모델을 이용해 캔(0)과 플라스틱(1)의 중심점 및 면적 반환 """
    # conf=0.7을 추가하여 모델이 70% 이상 확신할 때만 인식하도록 기준을 높입니다. (오탐지 획기적 감소)
    results = model(frame, stream=True, verbose=False, conf=0.65)
    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            box = boxes[0] 
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            target_name = "Can" if cls_id == 0 else "Plastic"
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            area = int((x2 - x1) * (y2 - y1))
            return cx, cy, area, True, target_name
    return 0, 0, 0, False, ""

def get_aruco_target(frame):
    """ OpenCV를 이용해 특정 ID의 ArUco 마커 중심점 및 면적 반환 """
    corners, ids, rejected = aruco_detector.detectMarkers(frame)
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == TARGET_BIN_ID:
                c = corners[i][0]
                cx = int(np.mean(c[:, 0]))
                cy = int(np.mean(c[:, 1]))
                area = int(cv2.contourArea(c))
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                return cx, cy, area, True
    return 0, 0, 0, False

def pickup_trash():
    """ 쓰레기 수거 하드웨어 제어 """
    print("[HW] Pickup started...")
    time.sleep(2)
    print("[HW] Pickup finished.")

def dropoff_trash():
    """ 쓰레기 배출 하드웨어 제어 """
    print("[HW] Dropoff started...")
    time.sleep(2)
    print("[HW] Dropoff finished.")

class DummyDrone:
    """ PC 테스트용 가상 드론 클래스 """
    def __enter__(self): return self
    def __exit__(self, t, v, tb): pass
    def set_mode(self, val): pass
    def set_althold(self, val): pass
    def arm(self, val): pass
    def set_sticks(self, **kwargs): pass
    def send(self): pass

def apply_pid(cx, cy, area, target_area):
    """ 데드존이 적용된 PID 제어값 계산 """
    error_x = cx - CENTER_X
    error_y = CENTER_Y - cy
    error_area = target_area - area

    # 데드존 이내이면 오차를 0으로 무시하여 드론의 흔들림 방지
    if abs(error_x) < DEADZONE_X: error_x = 0
    if abs(error_y) < DEADZONE_Y: error_y = 0

    yaw = max(-1.0, min(1.0, error_x * YAW_KP))
    pitch = max(-1.0, min(1.0, error_area * PITCH_KP))
    throttle = max(0.0, min(1.0, BASE_THROTTLE + (error_y * THROTTLE_KP)))
    return yaw, pitch, throttle

# ==========================================
# 5. 메인 루프 (Main Loop)
# ==========================================
def main():
    if not USE_SCREEN_CAPTURE:
        cap = cv2.VideoCapture(CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not cap.isOpened():
            return
    else:
        sct = mss.mss()
        monitor = sct.monitors[1]

    current_state = STATE_SEARCHING_TRASH
    drone_context = DummyDrone() if (TEST_MODE or Drone is None) else Drone("/dev/serial0")

    last_seen_time = 0
    last_seen_cx = CENTER_X

    try:
        with drone_context as drone:
            drone.set_mode(True)   
            drone.set_althold(True)
            drone.arm(True)        

            while True:
                if not USE_SCREEN_CAPTURE:
                    ret, frame = cap.read()
                    if not ret: break
                else:
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                
                roll, pitch, yaw, throttle = 0.0, 0.0, 0.0, BASE_THROTTLE
                
                # --- 상태 분기 ---
                if current_state in [STATE_SEARCHING_TRASH, STATE_TRACKING_TRASH]:
                    cx, cy, area, found, target_name = get_yolo_target(frame)
                    if found:
                        current_state = STATE_TRACKING_TRASH
                        last_seen_time = time.time()
                        last_seen_cx = cx
                        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                        
                        yaw, pitch, throttle = apply_pid(cx, cy, area, TARGET_TRASH_AREA)
                        
                        if area > TARGET_TRASH_AREA * 0.9:
                            current_state = STATE_PICKING_UP
                    else:
                        if time.time() - last_seen_time < GRACE_PERIOD:
                            # 1.5초 유예 시간 동안, 사라진 방향으로 계속 고개 돌리기
                            yaw = -0.15 if last_seen_cx < CENTER_X else 0.15
                        else:
                            # 유예 시간이 지나면 완전한 탐색 모드로 전환
                            current_state = STATE_SEARCHING_TRASH
                            yaw = 0.15

                elif current_state == STATE_PICKING_UP:
                    drone.set_sticks(roll=0, pitch=0, yaw=0, throttle=BASE_THROTTLE)
                    drone.send()
                    pickup_trash()
                    current_state = STATE_SEARCHING_BIN
                    last_seen_time = 0 

                elif current_state in [STATE_SEARCHING_BIN, STATE_TRACKING_BIN]:
                    cx, cy, area, found = get_aruco_target(frame)
                    if found:
                        current_state = STATE_TRACKING_BIN
                        last_seen_time = time.time()
                        last_seen_cx = cx
                        cv2.circle(frame, (cx, cy), 10, (255, 0, 0), -1)
                        
                        yaw, pitch, throttle = apply_pid(cx, cy, area, TARGET_BIN_AREA)
                        
                        if area > TARGET_BIN_AREA * 0.9:
                            current_state = STATE_DROPPING_OFF
                    else:
                        if time.time() - last_seen_time < GRACE_PERIOD:
                            yaw = -0.15 if last_seen_cx < CENTER_X else 0.15
                        else:
                            current_state = STATE_SEARCHING_BIN
                            yaw = 0.15

                elif current_state == STATE_DROPPING_OFF:
                    drone.set_sticks(roll=0, pitch=0, yaw=0, throttle=BASE_THROTTLE)
                    drone.send()
                    dropoff_trash()
                    current_state = STATE_DONE

                elif current_state == STATE_DONE:
                    pass
                # -----------------

                if current_state not in [STATE_PICKING_UP, STATE_DROPPING_OFF]:
                    drone.set_sticks(roll=roll, pitch=pitch, yaw=yaw, throttle=throttle)
                
                # 데드존 사각형 시각화 (화면 중앙 하얀 박스)
                cv2.rectangle(frame, (CENTER_X - DEADZONE_X, CENTER_Y - DEADZONE_Y), 
                                     (CENTER_X + DEADZONE_X, CENTER_Y + DEADZONE_Y), (255, 255, 255), 1)
                
                cv2.putText(frame, f"State: {current_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Drone Tracker", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord(' '):
                    if key == ord(' '): drone.arm(False)
                    break
                time.sleep(1.0 / TARGET_FPS)

    except KeyboardInterrupt:
        pass
    finally:
        if not USE_SCREEN_CAPTURE:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
