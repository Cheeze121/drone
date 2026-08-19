import time
import sys

try:
    from rpycrsf import Drone
except ImportError:
    print("[Error] rpycrsf library is not installed.")
    sys.exit()

SERIAL_PORT = "/dev/serial0" 
BASE_THROTTLE = 0.5  # Base throttle for hovering
TAKEOFF_DURATION = 10.0
MOVE_UP_DURATION = 2.0

def main():
    print("========================================")
    print("🚁 Drone Takeoff and Hover Test (rpycrsf)")
    print("⚠️ [WARNING] Please remove propellers before testing!")
    print("========================================\n")

    try:
        drone = Drone(SERIAL_PORT)
    except Exception as e:
        print(f"[Error] Failed to connect to drone: {e}")
        return

    # 1. Initialize (Safe State)
    print("[DEBUG] Initializing: All switches OFF, sticks at 0")
    drone.arm(False)
    drone.set_mode(False)
    drone.set_althold(False)
    drone.set_sticks(roll=0.0, pitch=0.0, yaw=0.0, throttle=0.0)
    drone.send()
    time.sleep(2)

    # 2. Turn on Aux switches for Altitude Hold
    print("[DEBUG] Mode and AltHold switches ON")
    drone.set_mode(True)
    drone.set_althold(True)
    drone.send()
    time.sleep(2)

    # 3. Pre-arm countdown (similar to auto_tracker.py)
    pre_arm_delay = 5
    print(f"[DEBUG] Arming in {pre_arm_delay}s...")
    for i in range(pre_arm_delay, 0, -1):
        print(f"[DEBUG] Takeoff in {i}s")
        time.sleep(1)

    # 4. Arm the drone
    print("[DEBUG] 🚨 Arming Drone (Arm Switch ON)")
    drone.arm(True)
    drone.send()
    time.sleep(1)

    # 5. Takeoff sequence (10 seconds)
    print(f"[DEBUG] Initiating takeoff sequence for {TAKEOFF_DURATION} seconds...")
    start_time = time.time()
    while time.time() - start_time < TAKEOFF_DURATION:
        # Use base throttle (0.5) to hover/takeoff
        drone.set_sticks(roll=0.0, pitch=0.0, yaw=0.0, throttle=BASE_THROTTLE)
        drone.send()
        time.sleep(0.1)
        
    print("[DEBUG] Takeoff sequence completed.")

    # 6. Move up slightly with throttle set to 0.2
    # (Setting throttle to 0.2 as requested)
    print("[DEBUG] Moving up slightly with throttle 0.2...")
    start_time = time.time()
    while time.time() - start_time < MOVE_UP_DURATION:
        drone.set_sticks(roll=0.0, pitch=0.0, yaw=0.0, throttle=0.2)
        drone.send()
        time.sleep(0.1)

    # 7. Continue hovering indefinitely
    print("[DEBUG] Transitioning to continuous hover mode.")
    print("[DEBUG] Hovering... (Press Ctrl+C to stop and disarm)")
    
    try:
        while True:
            # Revert to base throttle to maintain altitude
            drone.set_sticks(roll=0.0, pitch=0.0, yaw=0.0, throttle=BASE_THROTTLE)
            drone.send()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[DEBUG] Flight sequence interrupted by user. Landing...")

    # 8. Disarm and Exit
    print("[DEBUG] 🛑 Disarming Drone (Arm Switch OFF) and Exiting")
    drone.arm(False)
    drone.set_mode(False)
    drone.set_althold(False)
    drone.set_sticks(roll=0.0, pitch=0.0, yaw=0.0, throttle=0.0)
    drone.send()
    
    print("\n✅ Test completed safely.")

if __name__ == "__main__":
    main()