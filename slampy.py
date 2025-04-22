from ROVautonomous.Rov import Rov, FlightMode
from pymavlink import mavutil
import math
import slampy
import cv2
import numpy as np
import time

conn = mavutil.mavlink_connection('udpin:0.0.0.0:14550')
rov = Rov(conn)

config_files = {
    'mono': "mono_config.yaml",
    'stereo': "stereo_config.yaml",
    'mono_imu': "mono_imu_config.yaml",
    'stereo_imu': "stereo_imu_config.yaml",
}

slam_system = None
current_config = None
current_sensor_type = None

left_camera = cv2.VideoCapture(0)
right_camera = cv2.VideoCapture(1)

current_pos = {}
slam_running = False
last_depth_map = None

def initialize_slam(config_type):
    global slam_system, current_config, current_sensor_type

    if config_type not in config_files:
        print(f"Invalid configuration type: {config_type}")
        return False

    try:
        if slam_running:
            stop_slam()

        if config_type == 'mono':
            current_sensor_type = slampy.Sensor.MONOCULAR
        elif config_type == 'stereo':
            current_sensor_type = slampy.Sensor.STEREO
        elif config_type == 'mono_imu':
            current_sensor_type = slampy.Sensor.MONOCULAR_IMU
        elif config_type == 'stereo_imu':
            current_sensor_type = slampy.Sensor.STEREO_IMU

        slam_system = slampy.System(config_files[config_type], current_sensor_type)
        current_config = config_type
        print(f"SLAM initialized with {config_type} configuration")
        return True
    except Exception as e:
        print(f"Failed to initialize SLAM: {e}")
        return False

def get_imu_data():
    try:
        attitude = rov.state.attitude
        accel = rov.state.raw_imu
        ax = accel['xacc']
        ay = accel['yacc']
        az = accel['zacc']

        gx = accel['xgyro']
        gy = accel['ygyro']
        gz = accel['zgyro']

        timestamp = time.time()

        return {
            'timestamp': timestamp,
            'ax': ax, 'ay': ay, 'az': az,
            'gx': gx, 'gy': gy, 'gz': gz
        }
    except (KeyError, AttributeError):
        print("Failed to retrieve IMU data")
        return None

def process_mono_frame():
    ret, frame = left_camera.read()

    if not ret:
        print("Failed to capture frame")
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    timestamp = time.time()

    try:
        slam_system.process_image_mono(gray, timestamp)
        if slam_system.get_state() == slampy.State.OK:
            return slam_system.get_pose_to_target()
        else:
            return None
    except Exception as e:
        print(f"Error in mono processing: {e}")
        return None

def process_stereo_frame():
    ret_left, frame_left = left_camera.read()
    ret_right, frame_right = right_camera.read()

    if not ret_left or not ret_right:
        print("Failed to capture stereo frames")
        return None

    left_gray = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    timestamp = time.time()

    try:
        slam_system.process_image_stereo(left_gray, right_gray, timestamp)
        if slam_system.get_state() == slampy.State.OK:
            return slam_system.get_pose_to_target()
        else:
            return None
    except Exception as e:
        print(f"Error in stereo processing: {e}")
        return None

def process_mono_imu_frame():
    ret, frame = left_camera.read()

    if not ret:
        print("Failed to capture frame")
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    timestamp = time.time()

    imu_data = get_imu_data()
    if imu_data is None:
        print("Processing without IMU data")
        return process_mono_frame()

    imu_array = [
        imu_data['ax'], imu_data['ay'], imu_data['az'],
        imu_data['gx'], imu_data['gy'], imu_data['gz'],
        imu_data['timestamp']
    ]

    try:
        slam_system.process_image_imu_mono(gray, timestamp, imu_array)
        if slam_system.get_state() == slampy.State.OK:
            return slam_system.get_pose_to_target()
        else:
            return None
    except Exception as e:
        print(f"Error in mono-IMU processing: {e}")
        try:
            slam_system.process_image_mono(gray, timestamp)
            if slam_system.get_state() == slampy.State.OK:
                return slam_system.get_pose_to_target()
            else:
                return None
        except Exception as e2:
            print(f"Fallback processing failed: {e2}")
            return None

def process_stereo_imu_frame():
    ret_left, frame_left = left_camera.read()
    ret_right, frame_right = right_camera.read()

    if not ret_left or not ret_right:
        print("Failed to capture stereo frames")
        return None

    left_gray = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    timestamp = time.time()

    imu_data = get_imu_data()
    if imu_data is None:
        print("Processing without IMU data")
        return process_stereo_frame()

    imu_array = [
        imu_data['ax'], imu_data['ay'], imu_data['az'],
        imu_data['gx'], imu_data['gy'], imu_data['gz'],
        imu_data['timestamp']
    ]

    try:
        slam_system.process_image_imu_stereo(left_gray, right_gray, timestamp, imu_array)
        if slam_system.get_state() == slampy.State.OK:
            return slam_system.get_pose_to_target()
        else:
            return None
    except Exception as e:
        print(f"Error in stereo-IMU processing: {e}")
        try:
            slam_system.process_image_stereo(left_gray, right_gray, timestamp)
            if slam_system.get_state() == slampy.State.OK:
                return slam_system.get_pose_to_target()
            else:
                return None
        except Exception as e2:
            print(f"Fallback processing failed: {e2}")
            return None

def process_current_config():
    if current_config == 'mono':
        return process_mono_frame()
    elif current_config == 'stereo':
        return process_stereo_frame()
    elif current_config == 'mono_imu':
        return process_mono_imu_frame()
    elif current_config == 'stereo_imu':
        return process_stereo_imu_frame()
    else:
        print("No configuration selected")
        return None

def start_slam():
    global slam_running
    if slam_system is None:
        print("SLAM not initialized. Please select a configuration first.")
        return False

    slam_running = True
    print("SLAM system started")
    return True

def stop_slam():
    global slam_running
    if slam_running and slam_system is not None:
        slam_system.shutdown()
        slam_running = False
        print("SLAM system stopped")
        return True
    return False

def get_slam_position():
    if not slam_running:
        print("SLAM system is not running")
        return None

    pose = process_current_config()
    if pose is not None:
        position = pose[:3, 3]
        return position
    else:
        print("Failed to get position from SLAM")
        return None

def get_depth_map():
    global last_depth_map
    if not slam_running:
        print("SLAM system is not running")
        return None

    process_current_config()

    try:
        depth_map = slam_system.get_depth()
        if depth_map is not None:
            last_depth_map = depth_map
            return depth_map
        else:
            print("Failed to get depth map")
            return last_depth_map
    except Exception as e:
        print(f"Error getting depth map: {e}")
        return last_depth_map

def display_depth_map():
    depth = get_depth_map()
    if depth is not None:
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_display = np.uint8(depth_normalized)
        depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

        cv2.imshow('Depth Map', depth_colormap)
        cv2.waitKey(1)
    else:
        print("No depth map available")

while True:
    command = input('Please enter your Command [0-arm, 1-disarm, 2-MANUAL-mode, 3-GUIDED-mode, 4-get-current-pos, '
                   '5-set-waypoint, 6-getpos-from-distance, 7-flight-mode, 8-start-slam, 9-stop-slam, '
                   '10-get-slam-pos, 11-show-depth-map, 12-config-slam] >  ')

    if command == '0':
        rov.arm()
    elif command == '1':
        rov.disarm()
    elif command == '2':
        rov.change_flight_mode(FlightMode.MANUAL)
    elif command == '3':
        rov.change_flight_mode(FlightMode.GUIDED)
    elif command == '4':
        current_pos = rov.state.global_position_int
        print(current_pos)
    elif command == '5':
        rov.position_control.set_position_target_global_int(current_pos)
    elif command == '6':
        distance = float(input('Enter distance in m > '))
        attitude = rov.state.attitude
        yaw = float(attitude['yaw']) * 180 / math.pi
        pitch = float(attitude['pitch']) * 180 / math.pi

        x = distance * math.sin(yaw) * math.cos(pitch)
        y = distance * math.sin(pitch)
        z = distance * math.cos(yaw) * math.cos(pitch)

        print(f'x:{x}   y:{y}   z:{z}')
    elif command == '7':
        print(rov.state.flight_mode)
    elif command == '8':
        start_slam()
    elif command == '9':
        stop_slam()
    elif command == '10':
        slam_position = get_slam_position()
        if slam_position is not None:
            print(f"SLAM Position: x:{slam_position[0]}, y:{slam_position[1]}, z:{slam_position[2]}")
    elif command == '11':
        display_depth_map()
    elif command == '12':
        print("Available configurations:")
        for i, config in enumerate(config_files.keys()):
            print(f"{i+1}. {config}")

        config_choice = input("Select configuration (1-4): ")
        try:
            config_index = int(config_choice) - 1
            config_type = list(config_files.keys())[config_index]
            initialize_slam(config_type)
        except (ValueError, IndexError):
            print("Invalid selection")
    else:
        print("Invalid command")

cv2.destroyAllWindows()
stop_slam()
left_camera.release()
right_camera.release()
