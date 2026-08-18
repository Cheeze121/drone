import cv2
import time
import numpy as np
import mss

try:
    from rpycrsf import Drone
    import RPi.GPIO as GPIO
except ImportError:
    Drone = None
    GPIO = None

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
# 1-1. 초음파 센서 설정 (고도 유지)
# ==========================================
USE_ULTRASONIC = True      # 초음파 센서 고도 유지 기능 사용 여부
TRIG_PIN = 23              # 초음파 송신 핀 (BCM 번호)
ECHO_PIN = 24              # 초음파 수신 핀 (BCM 번호)
TARGET_ALTITUDE_CM = 50.0  # 유지할 목표 고도 (cm)
ALTITUDE_KP = 0.005        # 초음파 고도 제어 PID 비례 상수

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
    # conf=0.65로 약간 낮추되, 아래에서 크기와 비율로 안전 필터링 적용
    results = model(frame, stream=True, verbose=False, conf=0.65)
    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            box = boxes[0] 
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            w = int(x2 - x1)
            h = int(y2 - y1)
            area = w * h
            
            # 🚨 [안전 장치 1] 객체가 화면의 40% 이상을 차지할 정도로 너무 크면 무시 (사람 접근 방지)
            if area > (FRAME_WIDTH * FRAME_HEIGHT) * 0.4:
                continue
                
            # 🚨 [안전 장치 2] 세로가 가로보다 2배 이상 길면 사람(또는 다리)으로 간주하고 무시
            if h > w * 2.0:
                continue

            cls_id = int(box.cls[0].cpu().numpy())
            target_name = "Can" if cls_id == 0 else "Plastic"
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
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

def get_altitude():
    """ 초음파 센서로 바닥과의 거리를 측정하여 cm 단위로 반환 """
    if TEST_MODE or GPIO is None or not USE_ULTRASONIC:
        return TARGET_ALTITUDE_CM # 테스트 환경에서는 항상 목표 고도에 있다고 가정

    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)

    start_time = time.time()
    stop_time = time.time()
    timeout = start_time + 0.1 # 무한 루프 방지용 타임아웃 (0.1초)

    while GPIO.input(ECHO_PIN) == 0 and time.time() < timeout:
        start_time = time.time()

    while GPIO.input(ECHO_PIN) == 1 and time.time() < timeout:
        stop_time = time.time()

    elapsed = stop_time - start_time
    distance = (elapsed * 34300) / 2 # 왕복 거리이므로 2로 나눔

    # 거리가 너무 튀는 경우(2cm 미만, 400cm 초과) 방어 코드
    if distance < 2 or distance > 400:
        return TARGET_ALTITUDE_CM
    return distance

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

def apply_pid(cx, cy, area, target_area, current_altitude):
    """ 데드존 및 초음파 고도 제어가 적용된 PID 제어값 계산 """
    error_x = cx - CENTER_X
    error_y = CENTER_Y - cy
    error_area = target_area - area

    # 데드존 이내이면 오차를 0으로 무시하여 드론의 흔들림 방지
    if abs(error_x) < DEADZONE_X: error_x = 0
    if abs(error_y) < DEADZONE_Y: error_y = 0

    yaw = max(-1.0, min(1.0, error_x * YAW_KP))
    pitch = max(-1.0, min(1.0, error_area * PITCH_KP))
    
    if USE_ULTRASONIC:
        # 초음파 센서 사용 시: 바닥과의 거리를 기준으로 스로틀 제어
        error_altitude = TARGET_ALTITUDE_CM - current_altitude
        throttle = max(0.0, min(1.0, BASE_THROTTLE + (error_altitude * ALTITUDE_KP)))
    else:
        # 미사용 시: 카메라 Y축 기준으로 스로틀 제어
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

    # 초음파 센서 GPIO 초기화
    if USE_ULTRASONIC and not TEST_MODE and GPIO is not None:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(TRIG_PIN, GPIO.OUT)
        GPIO.setup(ECHO_PIN, GPIO.IN)
        GPIO.output(TRIG_PIN, False)
        print("✅ 초음파 센서 (HC-SR04) 핀 설정 완료")

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
                current_altitude = get_altitude() # 매 프레임마다 초음파 센서로 현재 고도 측정

                if current_state in [STATE_SEARCHING_TRASH, STATE_TRACKING_TRASH]:
                    cx, cy, area, found, target_name = get_yolo_target(frame)
                    if found:
                        current_state = STATE_TRACKING_TRASH
                        last_seen_time = time.time()
                        last_seen_cx = cx
                        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                        
                        yaw, pitch, throttle = apply_pid(cx, cy, area, TARGET_TRASH_AREA, current_altitude)
                        
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
                            
                        # 탐색 중에도 초음파 센서를 사용해 목표 고도를 일정하게 유지
                        if USE_ULTRASONIC:
                            error_altitude = TARGET_ALTITUDE_CM - current_altitude
                            throttle = max(0.0, min(1.0, BASE_THROTTLE + (error_altitude * ALTITUDE_KP)))

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
                        
                        yaw, pitch, throttle = apply_pid(cx, cy, area, TARGET_BIN_AREA, current_altitude)
                        
                        if area > TARGET_BIN_AREA * 0.9:
                            current_state = STATE_DROPPING_OFF
                    else:
                        if time.time() - last_seen_time < GRACE_PERIOD:
                            yaw = -0.15 if last_seen_cx < CENTER_X else 0.15
                        else:
                            current_state = STATE_SEARCHING_BIN
                            yaw = 0.15
                        
                        # 탐색 중에도 초음파 센서를 사용해 목표 고도를 일정하게 유지
                        if USE_ULTRASONIC:
                            error_altitude = TARGET_ALTITUDE_CM - current_altitude
                            throttle = max(0.0, min(1.0, BASE_THROTTLE + (error_altitude * ALTITUDE_KP)))

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
                if USE_ULTRASONIC:
                    cv2.putText(frame, f"Alt: {current_altitude:.1f}cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    
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
        if USE_ULTRASONIC and not TEST_MODE and GPIO is not None:
            GPIO.cleanup()

if __name__ == "__main__":
    main()
