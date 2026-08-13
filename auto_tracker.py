import cv2
import time
import numpy as np
from rpycrsf import Drone
from ultralytics import YOLO

# ==========================================
# 0. 설정 (Configuration)
# ==========================================
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

CENTER_X = FRAME_WIDTH // 2
CENTER_Y = FRAME_HEIGHT // 2

# 목표물과의 적정 거리 (바운딩 박스 넓이)
TARGET_TRASH_AREA = 25000  # 쓰레기 줍기 기준 넓이
TARGET_BIN_AREA = 15000    # 쓰레기통 도착 기준 넓이

# PID 제어 상수
YAW_KP = 0.0015
PITCH_KP = 0.00005
THROTTLE_KP = 0.002
BASE_THROTTLE = 0.5 

# ==========================================
# 1. 상태(State) 정의
# ==========================================
STATE_SEARCHING_TRASH = "탐색: 쓰레기 찾는 중..."
STATE_TRACKING_TRASH  = "추적: 쓰레기로 접근 중"
STATE_PICKING_UP      = "수거: 쓰레기 줍기 작동!"
STATE_SEARCHING_BIN   = "탐색: 쓰레기통(마커) 찾는 중..."
STATE_TRACKING_BIN    = "추적: 쓰레기통으로 접근 중"
STATE_DROPPING_OFF    = "배출: 쓰레기통에 버리기!"
STATE_DONE            = "임무 완료: 대기 중"

# ==========================================
# 2. AI 및 비전 모델 초기화
# ==========================================
print("🚀 YOLO 모델 로드 중...")
model = YOLO('best.pt')

print("🚀 ArUco 마커 디텍터 준비 중...")
# 4x4 크기의 마커 50개가 있는 사전을 사용 (ID 0 ~ 49)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
TARGET_BIN_ID = 0  # 쓰레기통에 붙일 마커의 ID

# ==========================================
# 3. 비전 탐지 함수들
# ==========================================
def get_yolo_target(frame):
    """YOLOv8로 캔(0) 또는 플라스틱(1) 찾기"""
    results = model(frame, stream=True, verbose=False)
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
    """OpenCV로 ArUco 마커 찾기"""
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
    if ids is not None:
        for i, marker_id in enumerate(ids):
            if marker_id[0] == TARGET_BIN_ID:
                c = corners[i][0]
                cx = int(np.mean(c[:, 0]))
                cy = int(np.mean(c[:, 1]))
                area = int(cv2.contourArea(c))
                
                # 화면에 마커 테두리 그리기
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                return cx, cy, area, True
    return 0, 0, 0, False

# ==========================================
# 4. 하드웨어 작동 뼈대 함수
# ==========================================
def pickup_trash():
    print("\n[하드웨어 제어] 🛠️ 집게 모터 작동 -> 쓰레기 줍기 시작...")
    time.sleep(2) # 2초 동안 줍는다고 가정
    print("[하드웨어 제어] ✅ 쓰레기 줍기 완료!\n")

def dropoff_trash():
    print("\n[하드웨어 제어] 🛠️ 집게 모터 풀기 -> 쓰레기통에 투하 시작...")
    time.sleep(2) # 2초 동안 놓는다고 가정
    print("[하드웨어 제어] ✅ 쓰레기 투하 완료!\n")

# ==========================================
# 5. 메인 제어 루프
# ==========================================
def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("❌ 에러: 카메라를 열 수 없습니다.")
        return

    current_state = STATE_SEARCHING_TRASH

    try:
        with Drone("/dev/serial0") as drone:
            print("✅ 드론 통신 포트 열림. 3초 후 시동을 켭니다.")
            time.sleep(3)
            drone.set_mode(True)   
            drone.set_althold(True)
            drone.arm(True)        
            print("🔥 시동 켜짐! 시나리오 시작.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                roll, pitch, yaw, throttle = 0.0, 0.0, 0.0, BASE_THROTTLE
                
                # ----------------------------------------------------
                # [상태 1 & 2] 쓰레기 찾기 및 접근
                # ----------------------------------------------------
                if current_state in [STATE_SEARCHING_TRASH, STATE_TRACKING_TRASH]:
                    cx, cy, area, found, target_name = get_yolo_target(frame)
                    if found:
                        current_state = STATE_TRACKING_TRASH
                        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                        
                        # PID 제어
                        yaw = max(-1.0, min(1.0, (cx - CENTER_X) * YAW_KP))
                        pitch = max(-1.0, min(1.0, (TARGET_TRASH_AREA - area) * PITCH_KP))
                        throttle = max(0.0, min(1.0, BASE_THROTTLE + ((CENTER_Y - cy) * THROTTLE_KP)))
                        
                        # 면적이 충분히 크면(가까워지면) 줍기 상태로 전환
                        if area > TARGET_TRASH_AREA * 0.9:
                            current_state = STATE_PICKING_UP
                    else:
                        current_state = STATE_SEARCHING_TRASH
                        # 타겟이 없으면 제자리에서 천천히 우회전하며 주변을 탐색합니다.
                        yaw = 0.15

                # ----------------------------------------------------
                # [상태 3] 쓰레기 줍기
                # ----------------------------------------------------
                elif current_state == STATE_PICKING_UP:
                    # 드론 정지
                    drone.set_sticks(roll=0, pitch=0, yaw=0, throttle=BASE_THROTTLE)
                    drone.send()
                    
                    cv2.putText(frame, current_state, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow("Drone Auto Tracker", frame)
                    cv2.waitKey(1)
                    
                    pickup_trash() # 뼈대 함수 호출
                    current_state = STATE_SEARCHING_BIN # 다 주웠으면 쓰레기통 찾기로 넘어감

                # ----------------------------------------------------
                # [상태 4 & 5] 쓰레기통(마커) 찾기 및 접근
                # ----------------------------------------------------
                elif current_state in [STATE_SEARCHING_BIN, STATE_TRACKING_BIN]:
                    cx, cy, area, found = get_aruco_target(frame)
                    if found:
                        current_state = STATE_TRACKING_BIN
                        cv2.circle(frame, (cx, cy), 10, (255, 0, 0), -1)
                        
                        # PID 제어
                        yaw = max(-1.0, min(1.0, (cx - CENTER_X) * YAW_KP))
                        pitch = max(-1.0, min(1.0, (TARGET_BIN_AREA - area) * PITCH_KP))
                        throttle = max(0.0, min(1.0, BASE_THROTTLE + ((CENTER_Y - cy) * THROTTLE_KP)))
                        
                        if area > TARGET_BIN_AREA * 0.9:
                            current_state = STATE_DROPPING_OFF
                    else:
                        current_state = STATE_SEARCHING_BIN
                        # 쓰레기통이 안 보이면 제자리에서 천천히 우회전하며 주변을 탐색합니다.
                        yaw = 0.15

                # ----------------------------------------------------
                # [상태 6] 쓰레기 버리기
                # ----------------------------------------------------
                elif current_state == STATE_DROPPING_OFF:
                    drone.set_sticks(roll=0, pitch=0, yaw=0, throttle=BASE_THROTTLE)
                    drone.send()
                    
                    cv2.putText(frame, current_state, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow("Drone Auto Tracker", frame)
                    cv2.waitKey(1)
                    
                    dropoff_trash()
                    current_state = STATE_DONE

                # ----------------------------------------------------
                # [상태 7] 임무 완료
                # ----------------------------------------------------
                elif current_state == STATE_DONE:
                    # 호버링 대기
                    pass

                # 드론 조종 신호 전송
                if current_state not in [STATE_PICKING_UP, STATE_DROPPING_OFF]:
                    drone.set_sticks(roll=roll, pitch=pitch, yaw=yaw, throttle=throttle)
                
                # 화면 출력 정보 갱신
                cv2.putText(frame, f"State: {current_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.line(frame, (CENTER_X, 0), (CENTER_X, FRAME_HEIGHT), (255, 255, 255), 1)
                cv2.line(frame, (0, CENTER_Y), (FRAME_WIDTH, CENTER_Y), (255, 255, 255), 1)
                
                cv2.imshow("Drone Auto Tracker", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    drone.arm(False)
                    drone.send()
                    print("\n🚨 긴급 정지!")
                    break
                    
                time.sleep(1.0 / TARGET_FPS)

    except KeyboardInterrupt:
        print("\n🛑 사용자 강제 종료.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
