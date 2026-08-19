import cv2
import time
import threading
import numpy as np
import mss

try:
    from rpycrsf import Drone
    import RPi.GPIO as GPIO
except ImportError:
    Drone = None
    GPIO = None

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

from ultralytics import YOLO

# ==========================================
# 1. Configuration
# ==========================================
TEST_MODE = False
USE_SCREEN_CAPTURE = False
USE_PICAMERA2 = Picamera2 is not None  # True when using a Raspberry Pi CSI camera (e.g. imx219)

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

# Advanced flight control settings (jitter prevention and grace period)
DEADZONE_X = 30     # Left/right deadzone (pixels)
DEADZONE_Y = 20     # Up/down deadzone (pixels)
GRACE_PERIOD = 1.5  # Grace period after losing the target (seconds)

# Stop-and-look search sweep: turn briefly, then hold still so the vision thread can get a
# clean (non-blurred) frame and finish a detection cycle before turning again.
SEARCH_TURN_DURATION = 0.4  # Time spent turning per sweep (seconds)
SEARCH_PAUSE_DURATION = 1.5 # Time spent holding still per sweep (seconds) - covers a full YOLO cycle (~1.1s)

PRE_ARM_DELAY = 10.0   # Wait this long after startup before arming/taking off (seconds)
LANDING_DURATION = 3.0 # Time to ramp throttle down to land on the trash (seconds)
PICKUP_DWELL_TIME = 10.0 # Time to stay landed while picking up the trash (seconds)

# ==========================================
# 1-1. Ultrasonic sensor settings (altitude hold)
# ==========================================
USE_ULTRASONIC = True      # Whether to use the ultrasonic altitude hold feature
TRIG_PIN = 23              # Ultrasonic trigger pin (BCM number)
ECHO_PIN = 24              # Ultrasonic echo pin (BCM number)
TARGET_ALTITUDE_CM = 50.0  # Target altitude to hold (cm)
ALTITUDE_KP = 0.005        # Proportional constant for ultrasonic altitude PID control

# ==========================================
# 2. Drone State (State Machine)
# ==========================================
STATE_SEARCHING_TRASH = "Search: Trash"
STATE_TRACKING_TRASH  = "Track: Trash"
STATE_PICKING_UP      = "Action: Pickup"
STATE_SEARCHING_BIN   = "Search: Bin"
STATE_TRACKING_BIN    = "Track: Bin"
STATE_DROPPING_OFF    = "Action: Dropoff"
STATE_DONE            = "Mission Complete"

# ==========================================
# 3. Vision AI (YOLO & ArUco) Initialization
# ==========================================
model = YOLO('best.pt')
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
TARGET_BIN_ID = 0

# ==========================================
# 4. Functions
# ==========================================
def get_yolo_target(frame):
    """ Returns the center point and area of a can(0) or plastic(1) using the YOLO model """
    # Lowered to conf=0.1, with extra safety filtering below by size and aspect ratio
    results = model(frame, stream=True, verbose=False, conf=0.1)
    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            box = boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            w = int(x2 - x1)
            h = int(y2 - y1)
            area = w * h

            # [Safety 1] Ignore if the object covers 40%+ of the frame (avoid approaching people)
            if area > (FRAME_WIDTH * FRAME_HEIGHT) * 0.4:
                continue

            # [Safety 2] Ignore if height is more than 2x width (likely a person or legs)
            if h > w * 2.0:
                continue

            cls_id = int(box.cls[0].cpu().numpy())
            target_name = "Can" if cls_id == 0 else "Plastic"
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            return cx, cy, area, True, target_name
    return 0, 0, 0, False, ""

def get_aruco_target(frame):
    """ Returns the center point and area of the target ArUco marker ID using OpenCV """
    corners, ids, rejected = aruco_detector.detectMarkers(frame)
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == TARGET_BIN_ID:
                c = corners[i][0]
                cx = int(np.mean(c[:, 0]))
                cy = int(np.mean(c[:, 1]))
                area = int(cv2.contourArea(c))
                return cx, cy, area, True
    return 0, 0, 0, False

# ==========================================
# 4-1. Background Vision Worker
# ==========================================
# YOLO inference alone takes ~1.1s per frame on this hardware. Running it inline in the
# control loop would cap altitude/attitude updates at <1Hz, which is far too slow to hover.
# This worker runs detection continuously in its own thread against whatever frame is newest,
# so the control loop below is never blocked waiting on it.
class SharedVision:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None
        self.mode = "trash"  # "trash" (YOLO) or "bin" (ArUco) requested by the control loop
        self.result_mode = None  # mode the current cx/cy/area/found were actually computed under
        self.cx, self.cy, self.area, self.found, self.name = 0, 0, 0, False, ""
        self.stop = False

def vision_worker(shared):
    while not shared.stop:
        with shared.lock:
            frame = shared.latest_frame
            mode = shared.mode
        if frame is None:
            time.sleep(0.01)
            continue

        if mode == "trash":
            cx, cy, area, found, name = get_yolo_target(frame)
        else:
            cx, cy, area, found = get_aruco_target(frame)
            name = ""

        with shared.lock:
            shared.cx, shared.cy, shared.area, shared.found, shared.name = cx, cy, area, found, name
            shared.result_mode = mode

_last_altitude = TARGET_ALTITUDE_CM
_last_ping_time = 0.0
ULTRASONIC_MIN_INTERVAL = 0.06  # HC-SR04 needs >=60ms between pings or echoes overlap and corrupt the reading
ULTRASONIC_TIMEOUT = 0.25       # Long enough to cover the sensor's own "no echo" window (measured ~193ms on this unit)

def get_altitude():
    """ Measures the distance to the ground with the ultrasonic sensor and returns it in cm """
    global _last_altitude, _last_ping_time

    if TEST_MODE or GPIO is None or not USE_ULTRASONIC:
        return TARGET_ALTITUDE_CM # Assume we're always at the target altitude in the test environment

    now = time.time()
    if now - _last_ping_time < ULTRASONIC_MIN_INTERVAL:
        return _last_altitude # Too soon since the last ping; reuse the last known-good reading
    _last_ping_time = now

    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)

    deadline = time.time() + ULTRASONIC_TIMEOUT

    start_time = time.time()
    while GPIO.input(ECHO_PIN) == 0:
        start_time = time.time()
        if start_time > deadline:
            return _last_altitude # ECHO never went high; sensor unresponsive

    stop_time = start_time
    while GPIO.input(ECHO_PIN) == 1:
        stop_time = time.time()
        if stop_time > deadline:
            return _last_altitude # No echo received within range (open air, or sensor fault)

    elapsed = stop_time - start_time
    distance = (elapsed * 34300) / 2 # Divide by 2 since it's a round trip distance

    # Guard against distance spikes (under 2cm or over 400cm)
    if 2 <= distance <= 400:
        _last_altitude = distance
    return _last_altitude

def pickup_trash():
    """ Controls the trash pickup hardware """
    print("[HW] Pickup started...")
    time.sleep(PICKUP_DWELL_TIME)
    print("[HW] Pickup finished.")

def dropoff_trash():
    """ Controls the trash dropoff hardware """
    print("[HW] Dropoff started...")
    time.sleep(2)
    print("[HW] Dropoff finished.")

class DummyDrone:
    """ Virtual drone class for PC testing """
    def __enter__(self): return self
    def __exit__(self, t, v, tb): pass
    def set_mode(self, val): pass
    def set_althold(self, val): pass
    def arm(self, val): pass
    def set_sticks(self, **kwargs): pass
    def send(self): pass

def apply_pid(cx, cy, area, target_area, current_altitude):
    """ Computes PID control values with deadzone and ultrasonic altitude control applied """
    error_x = cx - CENTER_X
    error_y = CENTER_Y - cy
    error_area = target_area - area

    # Zero out the error within the deadzone to prevent drone jitter
    if abs(error_x) < DEADZONE_X: error_x = 0
    if abs(error_y) < DEADZONE_Y: error_y = 0

    yaw = max(-1.0, min(1.0, error_x * YAW_KP))
    pitch = max(-1.0, min(1.0, error_area * PITCH_KP))

    if USE_ULTRASONIC:
        # When using the ultrasonic sensor: control throttle based on distance to the ground
        error_altitude = TARGET_ALTITUDE_CM - current_altitude
        throttle = max(0.0, min(1.0, BASE_THROTTLE + (error_altitude * ALTITUDE_KP)))
    else:
        # Otherwise: control throttle based on the camera's Y axis
        throttle = max(0.0, min(1.0, BASE_THROTTLE + (error_y * THROTTLE_KP)))

    return yaw, pitch, throttle

def search_sweep_yaw(phase_start):
    """ Turn briefly, then hold still, repeating - so detection gets a clean frame instead of a
    continuously blurred one. Returns (yaw, phase_start), where phase_start should be passed back
    in on the next call and reset to None whenever a new search sweep should start fresh. """
    if phase_start is None:
        phase_start = time.time()
    cycle = SEARCH_TURN_DURATION + SEARCH_PAUSE_DURATION
    phase_time = (time.time() - phase_start) % cycle
    yaw = 0.15 if phase_time < SEARCH_TURN_DURATION else 0.0
    return yaw, phase_start

def land_on_target(drone):
    """ Ramps throttle down to land in place, holding roll/pitch/yaw level """
    print("[FLIGHT] Landing...")
    steps = 30
    for i in range(steps, -1, -1):
        throttle = BASE_THROTTLE * (i / steps)
        drone.set_sticks(roll=0, pitch=0, yaw=0, throttle=throttle)
        drone.send()
        time.sleep(LANDING_DURATION / steps)
    drone.set_sticks(roll=0, pitch=0, yaw=0, throttle=0.0)
    drone.send()
    print("[FLIGHT] Landed.")

# ==========================================
# 5. Main Loop
# ==========================================
def main():
    if USE_SCREEN_CAPTURE:
        sct = mss.mss()
        monitor = sct.monitors[1]
    elif USE_PICAMERA2:
        picam2 = Picamera2()
        picam2_config = picam2.create_video_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
        )
        picam2.configure(picam2_config)
        picam2.start()
    else:
        cap = cv2.VideoCapture(CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not cap.isOpened():
            return

    current_state = STATE_SEARCHING_TRASH
    drone_context = DummyDrone() if (TEST_MODE or Drone is None) else Drone("/dev/serial0")
    if hasattr(drone_context, "ser"):
        # rpycrsf hardcodes write_timeout=0.1s, which is too tight while the
        # CPU-heavy YOLO thread is running and can starve the writer thread's
        # scheduling. Loosen it so transient scheduling delays don't trip
        # SerialTimeoutException spam (still caught internally either way).
        drone_context.ser.write_timeout = 1.0

    # Initialize ultrasonic sensor GPIO
    if USE_ULTRASONIC and not TEST_MODE and GPIO is not None:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(TRIG_PIN, GPIO.OUT)
        GPIO.setup(ECHO_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.output(TRIG_PIN, False)
        time.sleep(0.5) # Let the sensor settle before the first ping
        print("Ultrasonic sensor (HC-SR04) pin setup complete")

    last_seen_time = 0
    last_seen_cx = CENTER_X
    search_phase_start = None

    shared_vision = SharedVision()
    vision_thread = threading.Thread(target=vision_worker, args=(shared_vision,), daemon=True)
    vision_thread.start()

    try:
        with drone_context as drone:
            drone.set_mode(True)
            drone.set_althold(True)

            print(f"[FLIGHT] Arming in {PRE_ARM_DELAY:.0f}s...")
            countdown_start = time.time()
            while time.time() - countdown_start < PRE_ARM_DELAY:
                remaining = PRE_ARM_DELAY - (time.time() - countdown_start)
                print(f"[FLIGHT] Takeoff in {remaining:.0f}s")
                time.sleep(1)
            drone.arm(True)
            print("[FLIGHT] Armed. Taking off.")

            while True:
                if USE_SCREEN_CAPTURE:
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                elif USE_PICAMERA2:
                    frame = picam2.capture_array()
                else:
                    ret, frame = cap.read()
                    if not ret: break
                
                roll, pitch, yaw, throttle = 0.0, 0.0, 0.0, BASE_THROTTLE

                # Hand the newest frame to the background vision thread and pull its latest result.
                # Detection runs asynchronously (~1fps) so it never blocks this control loop.
                requested_mode = "trash" if current_state in [STATE_SEARCHING_TRASH, STATE_TRACKING_TRASH] else "bin"
                with shared_vision.lock:
                    shared_vision.latest_frame = frame.copy()
                    shared_vision.mode = requested_mode
                    cx, cy, area, found, target_name = (
                        shared_vision.cx, shared_vision.cy, shared_vision.area,
                        shared_vision.found, shared_vision.name,
                    )
                    # Ignore a result still left over from before we switched detection modes
                    if shared_vision.result_mode != requested_mode:
                        found = False

                # --- State branching ---
                current_altitude = get_altitude() # Measure current altitude with the ultrasonic sensor every frame

                if current_state in [STATE_SEARCHING_TRASH, STATE_TRACKING_TRASH]:
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
                            # During the grace period, keep turning toward the direction it disappeared
                            yaw = -0.15 if last_seen_cx < CENTER_X else 0.15
                            search_phase_start = None # Reset so a fresh sweep starts once the grace period ends
                        else:
                            # Once the grace period ends, switch to a stop-and-look search sweep
                            current_state = STATE_SEARCHING_TRASH
                            yaw, search_phase_start = search_sweep_yaw(search_phase_start)

                        # Keep holding the target altitude with the ultrasonic sensor even while searching
                        if USE_ULTRASONIC:
                            error_altitude = TARGET_ALTITUDE_CM - current_altitude
                            throttle = max(0.0, min(1.0, BASE_THROTTLE + (error_altitude * ALTITUDE_KP)))

                elif current_state == STATE_PICKING_UP:
                    land_on_target(drone)
                    drone.arm(False)
                    pickup_trash()
                    drone.arm(True)
                    current_state = STATE_SEARCHING_BIN
                    last_seen_time = 0

                elif current_state in [STATE_SEARCHING_BIN, STATE_TRACKING_BIN]:
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
                            search_phase_start = None # Reset so a fresh sweep starts once the grace period ends
                        else:
                            current_state = STATE_SEARCHING_BIN
                            yaw, search_phase_start = search_sweep_yaw(search_phase_start)

                        # Keep holding the target altitude with the ultrasonic sensor even while searching
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
                
                # Visualize the deadzone rectangle (white box in the center of the frame)
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
        shared_vision.stop = True
        vision_thread.join(timeout=2)
        if USE_PICAMERA2 and not USE_SCREEN_CAPTURE:
            picam2.stop()
        elif not USE_SCREEN_CAPTURE:
            cap.release()
        cv2.destroyAllWindows()
        if USE_ULTRASONIC and not TEST_MODE and GPIO is not None:
            GPIO.cleanup()

if __name__ == "__main__":
    main()
