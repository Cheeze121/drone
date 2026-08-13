import cv2
import time
from rpycrsf import Drone
import numpy as np
from ultralytics import YOLO

# 모델 로드 (학습된 가중치 파일 best.pt 사용)
model = YOLO('best.pt')

# ==========================================
# 설정 (Configuration)
# ==========================================
# 카메라 설정
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# 화면 중심 좌표 (드론이 타겟을 위치시키려는 목표점)
CENTER_X = FRAME_WIDTH // 2
CENTER_Y = FRAME_HEIGHT // 2

# 목표물과의 적정 거리 (바운딩 박스 넓이 기준)
# 너무 멀면 박스가 작아지고, 너무 가까우면 박스가 커집니다.
TARGET_BOX_AREA = 15000 

# PID 제어 상수 (P값: 비례 제어)
# 주의: 실제 비행 시 이 값들을 조금씩 조절(튜닝)해야 합니다.
# 값이 너무 크면 드론이 심하게 흔들리고, 작으면 반응이 너무 느립니다.
YAW_KP = 0.0015    # 좌우 회전 (목표물이 중심에서 벗어난 픽셀 당 회전량)
PITCH_KP = 0.00005 # 전후 이동 (목표물 크기 오차 당 전진/후진량)
THROTTLE_KP = 0.002 # 상승/하강 (목표물이 중심에서 위아래로 벗어난 픽셀 당 상승/하강량)

# ==========================================
# 객체 탐지 함수 (YOLO 연동부)
# ==========================================
def get_yolo_target(frame):
    """
    YOLOv8 모델을 사용하여 영상에서 타겟을 찾아 중심 좌표와 크기를 반환합니다.
    """
    # YOLO 모델 추론 (stream=True로 하면 메모리 효율이 좋습니다)
    results = model(frame, stream=True, verbose=False)
    
    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            # 탐지된 객체 중 첫 번째 객체를 추적합니다.
            box = boxes[0] 
            
            # 바운딩 박스 좌표 추출
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # 클래스 식별 (0: Can, 1: Plastic)
            cls_id = int(box.cls[0].cpu().numpy())
            target_name = "Can" if cls_id == 0 else "Plastic"
            
            # 중심점(cx, cy)과 면적(area) 계산
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            area = int((x2 - x1) * (y2 - y1))
            
            return cx, cy, area, True, target_name
            
    # 탐지된 객체가 없으면 False 반환
    return 0, 0, 0, False, ""

# ==========================================
# 메인 제어 루프
# ==========================================
def main():
    print("🚀 자율 주행 드론 프로그램을 시작합니다...")
    
    # 1. 카메라 초기화
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("❌ 에러: 카메라를 열 수 없습니다.")
        return

    print("✅ 카메라 연결 성공!")

    # 2. 드론 연결 및 시동
    try:
        # /dev/serial0 은 라즈베리파이의 기본 시리얼 포트입니다.
        with Drone("/dev/serial0") as drone:
            print("✅ 드론 통신 포트 열림. 3초 후 시동을 켭니다. (프로펠러 분리 필수!)")
            time.sleep(3)
            
            drone.set_mode(True)   # Angle 모드 (수평 유지) 켜기
            drone.set_althold(True) # 고도 유지 모드 켜기 (지원하는 경우)
            drone.arm(True)        # 시동 (모터 회전 시작)
            print("🔥 시동 켜짐! 자율 추적을 시작합니다.")

            # 기본 호버링 스틱 값 (드론에 따라 다름)
            base_throttle = 0.5 
            
            # 메인 루프
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ 카메라에서 프레임을 읽어올 수 없습니다.")
                    break
                
                # 영상에서 타겟 찾기
                cx, cy, area, found, target_name = get_yolo_target(frame)

                # 스틱 제어 변수 초기화 (매 프레임 0으로 초기화)
                roll, pitch, yaw, throttle = 0.0, 0.0, 0.0, base_throttle

                if found:
                    # [오차 계산]
                    error_x = cx - CENTER_X           # 양수: 타겟이 우측에 있음 -> 우회전(Yaw)
                    error_y = CENTER_Y - cy           # 양수: 타겟이 위쪽에 있음 -> 상승(Throttle)
                    error_area = TARGET_BOX_AREA - area # 양수: 타겟이 멀리 있음 -> 전진(Pitch)

                    # [PID 제어 적용 및 값 제한(-1.0 ~ 1.0)]
                    yaw = max(-1.0, min(1.0, error_x * YAW_KP))
                    pitch = max(-1.0, min(1.0, error_area * PITCH_KP))
                    
                    # 스로틀은 기본값에 오차를 더해줍니다. 0.0 ~ 1.0 사이 제한
                    throttle = max(0.0, min(1.0, base_throttle + (error_y * THROTTLE_KP)))
                    
                    # 화면에 추적 상태 및 객체 종류 표시
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                    cv2.putText(frame, f"Tracking: {target_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                else:
                    # 타겟을 잃어버렸을 때: 제자리 호버링 대기
                    yaw, pitch = 0.0, 0.0
                    throttle = base_throttle
                    cv2.putText(frame, f"Searching...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 조종 명령 전송 (값 업데이트)
                drone.set_sticks(roll=roll, pitch=pitch, yaw=yaw, throttle=throttle)
                
                # 현재 상태 화면 출력 (터미널)
                # print(f"Y:{yaw:.2f} | P:{pitch:.2f} | T:{throttle:.2f} | Target: {'O' if found else 'X'}", end='\r')

                # 화면에 십자선(중심점) 표시
                cv2.line(frame, (CENTER_X, 0), (CENTER_X, FRAME_HEIGHT), (255, 255, 255), 1)
                cv2.line(frame, (0, CENTER_Y), (FRAME_WIDTH, CENTER_Y), (255, 255, 255), 1)
                
                # 영상 출력
                cv2.imshow("Drone Auto Tracker", frame)
                
                # 'q' 키를 누르면 종료, 'space' 키를 누르면 즉시 시동 끄기 (긴급 정지)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n🛑 프로그램 종료를 요청했습니다.")
                    break
                elif key == ord(' '):
                    print("\n🚨 긴급 정지! 시동을 끕니다.")
                    drone.arm(False)
                    drone.send() # 즉시 전송
                    break
                    
                time.sleep(1.0 / TARGET_FPS)

    except KeyboardInterrupt:
        print("\n🛑 사용자가 강제로 종료했습니다 (Ctrl+C).")
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
    finally:
        # with 구문을 빠져나가면서 자동으로 Drone.close()가 호출되어 시동이 꺼지고 통신이 종료됩니다.
        cap.release()
        cv2.destroyAllWindows()
        print("✅ 자원을 반환하고 안전하게 종료되었습니다.")

if __name__ == "__main__":
    main()
