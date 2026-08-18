import time
import sys
    
try:
    from rpycrsf import Drone
except ImportError:
    print("❌ rpycrsf 라이브러리가 설치되지 않았습니다. (터미널에 pip install rpycrsf 입력)")
    sys.exit()
    
# 포트 설정 (환경에 맞춰 변경하세요)
SERIAL_PORT = "/dev/serial0" 
    
def main():
        print("========================================")
        print("🚁 드론 자동 제어 시퀀스 테스트 (rpycrsf)")
        print("⚠️ [경고] 반드시 프로펠러를 분리하고 진행하세요!")
        print("========================================\n")
    
        try:
            drone = Drone(SERIAL_PORT)
        except Exception as e:
            print(f"❌ 드론 연결 실패: {e}")
            return
    
        # 1. 초기화 (안전 상태)
        print("[0초] 초기화: 모든 스위치 OFF, 스틱 0")
        drone.arm(False)
        drone.set_mode(False)
        drone.set_althold(False)
        drone.set_sticks(roll=0.0, pitch=0.0, yaw=0.0, throttle=0.0)
        drone.send()
        time.sleep(2) # 2초 대기
    
        # 2. 보조 스위치 켜기
        print("[2초] Mode 및 AltHold 스위치 ON")
        drone.set_mode(True)
        drone.set_althold(True)
        drone.send()
        time.sleep(2)
    
        # 3. 시동 켜기
        print("[4초] 🚨 시동(Arm) ON")
        drone.arm(True)
        drone.send()
        time.sleep(2)
    
        # 4. 스로틀 테스트
        print("[6초] Throttle(스로틀) 20% 상승")
        drone.set_sticks(roll=0.0, pitch=0.0, yaw=0.0, throttle=0.2)
        drone.send()
        time.sleep(2)
    
        # 5. 피치(Pitch) 테스트
        print("[8초] Pitch(피치) 전진 방향으로 30%")
        drone.set_sticks(roll=0.0, pitch=0.3, yaw=0.0, throttle=0.2)
        drone.send()
        time.sleep(2)
    
        # 6. 롤(Roll) 테스트
        print("[10초] Roll(롤) 우측 방향으로 30%")
        drone.set_sticks(roll=0.3, pitch=0.0, yaw=0.0, throttle=0.2)
        drone.send()
        time.sleep(2)
    
        # 7. 요(Yaw) 테스트
        print("[12초] Yaw(요) 우회전 방향으로 30%")
        drone.set_sticks(roll=0.0, pitch=0.0, yaw=0.3, throttle=0.2)
        drone.send()
        time.sleep(2)
    
        # 8. 스틱 원위치
        print("[14초] 모든 방향 스틱 중앙 정렬 (스로틀만 유지)")
        drone.set_sticks(roll=0.0, pitch=0.0, yaw=0.0, throttle=0.1)
        drone.send()
        time.sleep(2)
    
        # 9. 테스트 종료
        print("[16초] 🛑 시동(Arm) OFF 및 프로그램 종료")
        drone.arm(False)
        drone.set_mode(False)
        drone.set_althold(False)
        drone.set_sticks(roll=0.0, pitch=0.0, yaw=0.0, throttle=0.0)
        drone.send()
        
        print("\n✅ 모든 자동화 테스트가 안전하게 완료되었습니다.")

if __name__ == "__main__":
        main()