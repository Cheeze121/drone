import time
import sys
    
try:
        from rpycrsf import Drone
except ImportError:
        print("[Error] rpycrsf library is not installed.")
        sys.exit()
    
SERIAL_PORT = "/dev/serial0" 
    
def main():
        print("========================================")
        print("🚁 Drone Auto Control Sequence Test (rpycrsf)")
        print("⚠️ [WARNING] Please remove propellers before testing!")
        print("========================================\n")
    
        try:
            drone = Drone(SERIAL_PORT)
        except Exception as e:
            print(f"[Error] Failed to connect to drone: {e}")
            return
        # 1. Initialize (Safe State)
    
        print("[0s] Initializing: All switches OFF, sticks at 0")
        drone.arm(False)
        drone.set_mode(False)
        drone.set_althold(False)
        drone.set_sticks(roll=0.0, pitch=0.0, yaw=0.0, throttle=0.0)
        drone.send()
        time.sleep(2)
    
        # 2. Turn on Aux switches
        print("[2s] Mode and AltHold switches ON")
        drone.set_mode(True)
        drone.set_althold(True)
        drone.send()
        time.sleep(2)
    
        # 3. Arm the drone
        drone.arm(True)
        print("[4s] 🚨 Arming Drone (Arm Switch ON)")
        drone.send()
        time.sleep(2)
    
        # 4. Throttle Test (Modified to 1.0)
        print("[6s] Testing Throttle: UP to 1.0 (100%)")
        drone.set_sticks(roll=0.0, pitch=0.0, yaw=0.0, throttle=1.0)
        drone.send()
        time.sleep(2)
    
        # 5. Pitch Test
        print("[8s] Testing Pitch: Forward 30%")
        drone.set_sticks(roll=0.0, pitch=0.3, yaw=0.0, throttle=1.0)
        drone.send()
        time.sleep(2)
    
        # 6. Roll Test
        print("[10s] Testing Roll: Right 30%")
        drone.set_sticks(roll=0.3, pitch=0.0, yaw=0.0, throttle=1.0)
        drone.send()
        time.sleep(2)
    
        # 7. Yaw Test
        print("[12s] Testing Yaw: Right rotate 30%")
        drone.set_sticks(roll=0.0, pitch=0.0, yaw=0.3, throttle=1.0)
        drone.send()
        time.sleep(2)
    
        # 8. Reset Sticks
        print("[14s] Resetting all sticks to center (Throttle maintained at 0.1)")
        drone.set_sticks(roll=0.0, pitch=0.0, yaw=0.0, throttle=0.1)
        drone.send()
        time.sleep(2)

        # 9. Disarm and Exit
        print("[16s] 🛑 Disarming Drone (Arm Switch OFF) and Exiting")
        drone.arm(False)
        drone.set_mode(False)
        drone.set_althold(False)
        drone.set_sticks(roll=0.0, pitch=0.0, yaw=0.0, throttle=0.0)
        drone.send()
        
        print("\n✅ All automated tests completed safely.")

if __name__ == "__main__":
        main()