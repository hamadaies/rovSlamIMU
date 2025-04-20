import math
import time
import cv2
import numpy as np
import torch

from pymavlink import mavutil
from ROVautonomous.Rov import Rov, FlightMode

from slam.ekf import EKFSLAM
from utils.vis import visualize_map

def request_imu_stream(conn, frequency_hz=50):
    msg = conn.mav.command_long_encode(
        conn.target_system,
        conn.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        mavutil.mavlink.MAVLINK_MSG_ID_RAW_IMU,
        1e6 / frequency_hz,
        0, 0, 0, 0, 0
    )
    conn.send(msg)

def get_imu_data(rov):
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

def get_camera_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

def extract_features(frame, max_features=100):
    feature_params = dict(
        maxCorners=max_features,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=7
    )
    corners = cv2.goodFeaturesToTrack(frame, mask=None, **feature_params)
    if corners is None:
        return None
    
    stereo_features = np.zeros((4, len(corners)))
    for i, corner in enumerate(corners):
        x, y = corner.ravel()
        stereo_features[0, i] = x  # ul
        stereo_features[1, i] = y  # vl
        stereo_features[2, i] = x  # ur (since we're simulating stereo)
        stereo_features[3, i] = y  # vr
    
    return stereo_features

def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return None
    
    ret, _ = cap.read()
    if not ret:
        print("Error: Camera initialized but unable to capture frames.")
        cap.release()
        return None
    
    return cap

def run_vi_slam(rov, slam, cap, start_time):
    print("[VI-SLAM] Running live stream with IMU + Camera...")
    print("Press 'q' in visualization window or Ctrl+C to exit")
    
    K = torch.tensor([
        [525.0, 0, 320.0],
        [0, 525.0, 240.0],
        [0, 0, 1.0]
    ]).double().to(slam.device)
    
    baseline = torch.tensor(0.1).double().to(slam.device)
    
    imu_T_cam = torch.eye(4).double().to(slam.device)
    
    prev_time = time.time() - start_time
    poses = []
    
    try:
        while True:
            frame = get_camera_frame(cap)
            if frame is None:
                print("Waiting for valid camera frame...")
                time.sleep(0.1)
                continue
            
            accel, gyro = get_imu_data(rov)
            if accel is None or gyro is None:
                print("Waiting for valid IMU data...")
                time.sleep(0.1)
                continue
            
            current_time = time.time() - start_time
            dt = current_time - prev_time
            
            if dt < 0.001:
                continue
            
            u = np.concatenate((accel, gyro))
            u_tensor = torch.from_numpy(u).float().to(slam.device)
            
            slam.predict(u_tensor[None, :], dt)
            
            features = extract_features(frame)
            if features is not None:
                features_tensor = torch.from_numpy(features).double().to(slam.device)
                slam.update(features_tensor, K, baseline, imu_T_cam)
            
            pose = slam.get_pose(numpy=True)
            poses.append(pose)
            
            print(f"SLAM Pose → x:{pose[0, 0, 3]:.2f}, y:{pose[0, 1, 3]:.2f}, z:{pose[0, 2, 3]:.2f}")
            
            vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            if features is not None:
                for i in range(features.shape[1]):
                    x, y = int(features[0, i]), int(features[1, i])
                    cv2.circle(vis_frame, (x, y), 3, (0, 255, 0), -1)
            
            cv2.imshow("Camera View", vis_frame)
            
            if len(poses) > 1:
                trajectory = np.array(poses).squeeze(1).transpose(1, 2, 0)
                landmarks = slam.get_landmarks().cpu().numpy()
                
                plt.figure(figsize=(10, 8))
                plt.subplot(2, 1, 1)
                plt.scatter(landmarks[0, :], landmarks[2, :], c='r', marker='.', s=5)
                plt.plot(trajectory[0, 3, :], trajectory[2, 3, :], 'b-', linewidth=1)
                plt.grid(True)
                plt.axis('equal')
                plt.title('Top View (XZ)')
                
                plt.subplot(2, 1, 2)
                plt.scatter(landmarks[0, :], landmarks[1, :], c='r', marker='.', s=5)
                plt.plot(trajectory[0, 3, :], trajectory[1, 3, :], 'b-', linewidth=1)
                plt.grid(True)
                plt.axis('equal')
                plt.title('Side View (XY)')
                
                plt.tight_layout()
                plt.savefig('/tmp/slam_map.png')
                plt.close()
                
                map_img = cv2.imread('/tmp/slam_map.png')
                if map_img is not None:
                    cv2.imshow("SLAM Map", map_img)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
            prev_time = current_time
            
    except KeyboardInterrupt:
        print("\nSLAM process interrupted by user")
    except Exception as e:
        print(f"Error during SLAM operation: {e}")
    finally:
        cv2.destroyAllWindows()

def main():
    try:
        conn = mavutil.mavlink_connection("udpin:0.0.0.0:14550")
        conn.wait_heartbeat()
        rov = Rov(conn)
        request_imu_stream(conn)

        device = 'cpu'
        imu_T_cam = torch.eye(4).double().to(device)
        
        slam = EKFSLAM(n_landmarks=100, imu_T_cam=imu_T_cam, device=device)
        
        cap = initialize_camera()
        if cap is None:
            print("Error: Unable to initialize camera. VI-SLAM will not be available.")
            return
        
        start_time = time.time()
        current_pos = {}

        while True:
            command = input(
                "\nEnter Command [0-arm, 1-disarm, 2-MANUAL, 3-GUIDED, 4-get-pos, 5-set-waypoint,"
                " 6-xyz-from-distance, 7-flight-mode, 8-IMU, 9-run-VI-SLAM, q-quit] > "
            )

            if command.lower() == "q":
                break
                
            elif command == "0":
                rov.arm()
                
            elif command == "1":
                rov.disarm()
                
            elif command == "2":
                rov.change_flight_mode(FlightMode.MANUAL)
                
            elif command == "3":
                rov.change_flight_mode(FlightMode.GUIDED)
                
            elif command == "4":
                current_pos = rov.state.global_position_int
                print(current_pos)
                
            elif command == "5":
                if current_pos:
                    rov.position_control.set_position_target_global_int(current_pos)
                else:
                    print("Use command 4 first.")
                    
            elif command == "6":
                try:
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
                except ValueError:
                    print("Invalid input.")
                    
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
                run_vi_slam(rov, slam, cap, start_time)
                
            else:
                print("Unknown command.")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
