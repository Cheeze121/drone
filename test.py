import time
import sys

try:
    from rpycrsf import Drone
except ImportError:
    print("[Error] rpycrsf library is not installed.")
    sys.exit()

SERIAL_PORT = "/dev/serial0" 
BASE_THROTTLE = 0.5  # Base throttle for hovering

def main():
    print("========================================")
    print("🚁 Drone Hover and FORCE KILL Test (rpycrsf)")
    print("⚠️ [WARNING] This will forcefully drop the drone after 5s!")
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

    # 3. Pre-arm countdown
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

    # 5. Hover sequence (5 seconds)
    hover_duration = 5.0
    print(f"[DEBUG] Hovering for {hover_duration} seconds...")
    start_time = time.time()
    while time.time() - start_time < hover_duration:
        drone.set_sticks(roll=0.0, pitch=0.0, yaw=0.0, throttle=BASE_THROTTLE)
        drone.send()
        time.sleep(0.1)
        
    print("[DEBUG] Hover sequence completed. Initiating forceful shutdown!")

    # 6. Forcibly Disarm and Power Off
    print("[DEBUG] 🛑 KILL SWITCH ACTIVATED: FORCIBLY SHUTTING DOWN DRONE")
    drone.arm(False)
    drone.set_mode(False)
    drone.set_althold(False)
    drone.set_sticks(roll=0.0, pitch=0.0, yaw=0.0, throttle=0.0)
    drone.send()
    
    print("\n✅ Drone has been forcibly powered off (dropped).")

if __name__ == "__main__":
    main()