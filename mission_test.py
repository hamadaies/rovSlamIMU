import math
import time
import cv2
import numpy as np
from pymavlink import mavutil
from ROVautonomous.Rov import Rov, FlightMode
from slam.ekf import EKFSLAM
from utils.vis import visualize_trajectory_map_2d
import torch

conn = mavutil.mavlink_connection("udpin:0.0.0.0:14550")
rov = Rov(conn)

# Request IMU stream
msg = conn.mav.command_long_encode(
    conn.target_system,
    conn.target_component,
    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
    0,
    mavutil.mavlink.MAVLINK_MSG_ID_RAW_IMU,
    20000,
    0,
    0,
    0,
    0,
    0,  # 50Hz
)
conn.send(msg)

# Camera intrinsic parameters
K = torch.tensor([
    [400, 0, 320],
    [0, 400, 240],
    [0, 0, 1]
]).double()
baseline = 0.1  # 10cm baseline for stereo

slam = EKFSLAM(n_landmarks=50, imu_T_cam=torch.eye(4).double())

cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

start_time = time.time()
last_time = start_time
current_pos = {}
poses_history = torch.zeros((4, 4, 1000)).double()
pose_count = 0

def get_imu_data():
    imu = rov.state.raw_imu
    if not imu:
        return None, None

    accel = np.array([
        imu["xacc"] / 1000.0,
        imu["yacc"] / 1000.0,
        imu["zacc"] / 1000.0,
    ])

    gyro = np.array([
        imu["xgyro"] / 1000.0,
        imu["ygyro"] / 1000.0,
        imu["zgyro"] / 1000.0,
    ])

    return accel, gyro

def get_stereo_frames():
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        return None, None

    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    return gray_left, gray_right

def extract_features(img_left, img_right):
    orb = cv2.ORB_create(nfeatures=100)

    kp_left, des_left = orb.detectAndCompute(img_left, None)
    kp_right, des_right = orb.detectAndCompute(img_right, None)

    if des_left is None or des_right is None or len(kp_left) < 10 or len(kp_right) < 10:
        return torch.full((4, 50), -1).double()

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_left, des_right)

    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:min(50, len(matches))]

    features = torch.full((4, 50), -1).double()

    for i, match in enumerate(matches):
        if i >= 50:
            break

        left_idx = match.queryIdx
        right_idx = match.trainIdx

        features[0, i] = kp_left[left_idx].pt[0]    # ul
        features[1, i] = kp_left[left_idx].pt[1]    # vl
        features[2, i] = kp_right[right_idx].pt[0]  # ur
        features[3, i] = kp_right[right_idx].pt[1]  # vr

    return features

def visualize_map(slam_system):
    landmarks = slam_system.get_landmarks().cpu().numpy()
    pose = slam_system.get_pose(numpy=True)

    global pose_count, poses_history
    if pose_count < poses_history.shape[2]:
        poses_history[:, :, pose_count] = torch.tensor(pose).double()
        pose_count += 1

    visualize_trajectory_map_2d(
        poses_history[:, :, :pose_count].cpu().numpy(),
        landmarks,
        path_name="ROV Trajectory"
    )

while True:
    command = input(
        "\nEnter Command [0-arm, 1-disarm, 2-MANUAL, 3-GUIDED, 4-get-pos, 5-set-waypoint,"
        " 6-xyz-from-distance, 7-flight-mode, 8-IMU, 9-run-VI-SLAM, q-quit] > "
    )

    if command == "0":
        rov.arm()
    elif command == "1":
        rov.disarm()
    elif command == "2":
        rov.change_flight_mode(FlightMode.MANUAL.value)
    elif command == "3":
        rov.change_flight_mode(FlightMode.GUIDED.value)
    elif command == "4":
        current_pos = rov.state.global_position_int
        print(current_pos)
    elif command == "5":
        if current_pos:
            rov.position_control.set_position_target_global_int(current_pos)
        else:
            print("Use command 4 first.")
    elif command == "6":
        distance = float(input("Enter distance (m): "))
        attitude = rov.state.attitude
        if attitude:
            yaw = float(attitude["yaw"])
            pitch = float(attitude["pitch"])
            x = distance * math.cos(yaw) * math.cos(pitch)
            y = distance * math.sin(yaw) * math.cos(pitch)
            z = -distance * math.sin(pitch)
            print(f"x: {x:.2f}, y: {y:.2f}, z: {z:.2f}")
        else:
            print("No attitude data.")
    elif command == "7":
        print(rov.state.flight_mode)
    elif command == "8":
        imu = rov.state.raw_imu
        if imu:
            print(f"RAW_IMU:")
            print(f"Accel: X={imu['xacc']}, Y={imu['yacc']}, Z={imu['zacc']}")
            print(f"Gyro:  X={imu['xgyro']}, Y={imu['ygyro']}, Z={imu['zgyro']}")
            print(f"Mag:   X={imu['xmag']}, Y={imu['ymag']}, Z={imu['zmag']}")
        else:
            print("No IMU data.")
    elif command == "9":
        print("[VI-SLAM] Running live stream with IMU + Camera...")
        try:
            running = True
            while running:
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time

                accel, gyro = get_imu_data()
                if accel is None or gyro is None:
                    print("Waiting for valid IMU data...")
                    time.sleep(0.1)
                    continue

                u = torch.tensor(gyro).double()

                slam.predict(u, dt)

                left_frame, right_frame = get_stereo_frames()
                if left_frame is None or right_frame is None:
                    print("Waiting for valid camera frames...")
                    time.sleep(0.1)
                    continu

                features = extract_features(left_frame, right_frame)

                slam.update(features, K, baseline, torch.eye(4).double())

                pose = slam.get_pose()
                pos = pose[0:3, 3].cpu().numpy()
                print(f"SLAM Pose → x:{pos[0]:.2f}, y:{pos[1]:.2f}, z:{pos[2]:.2f}")

                if pose_count % 10 == 0:
                    visualize_map(slam)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("SLAM visualization stopped by user")
    elif command == "q" or command == "quit":
        break
    else:
        print("Unknown command.")

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
