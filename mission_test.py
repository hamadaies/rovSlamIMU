from ROVautonomous.Rov import Rov, FlightMode
from pymavlink import mavutil
import math


conn = mavutil.mavlink_connection("udpin:0.0.0.0:14550")
rov = Rov(conn)

msg = conn.mav.command_long_encode(
    conn.target_system,
    conn.target_component,
    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
    0,
    mavutil.mavlink.MAVLINK_MSG_ID_RAW_IMU,  # Use MAVLINK_MSG_ID_SCALED_IMU2 if needed
    20000,  # Interval in microseconds (20000 µs = 50 Hz)
    0,
    0,
    0,
    0,
    0,
)
conn.send(msg)

current_pos = {}
latest_imu_data = None

while True:
    command = input(
        "Please enter your Command [0-arm, 1-disarm, 2-MANUAL-mode, 3-GUIDED-mode, 4-get-current-pos, 5-set-waypoint, 6-getpos-from-distance, 7-flight-mode] >  "
    )

    if command == "0":
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
        rov.position_control.set_position_target_global_int(current_pos)
    elif command == "6":
        distance = float(input("Enter distance in m > "))
        attitude = rov.state.attitude
        yaw = float(attitude["yaw"]) * 180 / math.pi
        pitch = float(attitude["pitch"]) * 180 / math.pi

        x = distance * math.sin(yaw) * math.cos(pitch)
        y = distance * math.sin(pitch)
        z = distance * math.cos(yaw) * math.cos(pitch)

        print(f"x:{x}   y:{y}   z:{z}")
    elif command == "7":
        print(rov.state.flight_mode)
    elif command == "8":
        latest_imu_data = rov.state.raw_imu  # or rov.state.scaled_imu2
        if latest_imu_data:
            print(f"RAW_IMU Data:")
            print(
                f"Accel: X={latest_imu_data['xacc']}, Y={latest_imu_data['yacc']}, Z={latest_imu_data['zacc']}"
            )
            print(
                f"Gyro: X={latest_imu_data['xgyro']}, Y={latest_imu_data['ygyro']}, Z={latest_imu_data['zgyro']}"
            )
            print(
                f"Mag: X={latest_imu_data['xmag']}, Y={latest_imu_data['ymag']}, Z={latest_imu_data['zmag']}"
            )
        else:
            print("No IMU data received.")
    else:
        print("Unknown command.")
