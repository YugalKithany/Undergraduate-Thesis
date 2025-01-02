import airsim
import numpy as np
import time
import os
import cv2
from controller_pid import PIDController
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import math
import json
from datetime import datetime

WAYPOINTS = [[0.0, 2.0, 2.0199999809265137], [1.5999999046325684, 10.800000190734863, 2.0199999809265137], 
             [8.887084007263184, 18.478761672973633, 2.0199999809265137], [18.74375343322754, 22.20650863647461, 2.0199999809265137],
             [30.04375457763672, 22.20648956298828, 2.0199999809265137], [39.04375457763672, 19.206478118896484, 2.0199999809265137],
            [45.74375534057617, 11.706478118896484, 2.0199999809265137], [45.74375534057617, 2.2064781188964844, 2.0199999809265137], 
            [40.343753814697266, -4.793521404266357, 2.0199999809265137], [30.74375343322754, -7.893521785736084,2.0199999809265137],
            [18.54375457763672, -7.893521785736084, 2.0199999809265137], [9.543754577636719, -5.093521595001221, 2.0199999809265137]]

WAYPOINTS_ANGLES = [
    15, 30, 45,60,75,90,120,135,150,165,180,195]

class simulation():
    def __init__(self, totalcount=50):
        self.lead = "Drone_L"
        self.client1 = airsim.MultirotorClient()
        self.client1.confirmConnection()
        self.client1.enableApiControl(True,self.lead)
        self.client1.armDisarm(True, self.lead)
        self.client1.takeoffAsync(30.0, self.lead).join()

# Coordinates of start and end gates
START_POS = [6.3, 81, -43]
END_POS = [-24.641, 9.132, 2.120]
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
# PID controller setup
gain_x = [3, 0, 8.0]  # Reduced from [5, 0, 10.0]
gain_y = [3, 0, 8.0]  # Reduced from [5, 0, 10.0]
gain_z = [1, 0, 5.0]  # Keep the same
pid = PIDController(gain_x=gain_x, gain_y=gain_y, gain_z=gain_z)

def move_by_waypoints():
    global drone_path
    drone_path = []
    airsim_waypoints = [airsim.Vector3r(wp[0], wp[1], wp[2]) for wp in WAYPOINTS]
    client.moveOnPathAsync(airsim_waypoints, velocity=5, drivetrain=airsim.DrivetrainType.ForwardOnly, 
                           yaw_mode=airsim.YawMode(False, 0), lookahead=-1, adaptive_lookahead=1)
    start_time = time.time()
    while time.time() - start_time < len(WAYPOINTS) * 5:
        pos = client.getMultirotorState().kinematics_estimated.position
        drone_path.append([pos.x_val, pos.y_val, pos.z_val])
        time.sleep(1)  # Save coordinates every second

    # for wp in WAYPOINTS:
    #     start_time = time.time()
    #     # client.moveToPositionAsync(wp[0], wp[1], wp[2], 5).join()
        # while time.time() - start_time < 5:  
        #     pos = client.getMultirotorState().kinematics_estimated.position
        #     drone_path.append([pos.x_val, pos.y_val, pos.z_val])
        #     time.sleep(1)  # Save coordinates every second


    print("Drone reached all waypoints.")


def calculate_yaw_angle(current_pos, target_pos):
    # Calculate the yaw angle based on the direction vector to the waypoint
    delta_x = target_pos[0] - current_pos[0]
    delta_y = target_pos[1] - current_pos[1]
    yaw = math.atan2(delta_y, delta_x)  # Calculate yaw in radians
    yaw_deg = math.degrees(yaw)         # Convert to degrees
    return yaw_deg




def state_based_pid_control(pidC):
    global drone_path
    drone_path = []
    gate_clearance_positions = []  # Store positions where gates were cleared
    collision_count = 0
    run_start_time = time.time()
    for i, wp in enumerate(WAYPOINTS):
        print(f"Target waypoint: {wp}")
        current_pos = client.getMultirotorState().kinematics_estimated.position
        pidC.update_setpoint(wp)
        
        final_approach_velocity = [0, 0, 0]
        stuck_timestamp = time.time()
        while not np.allclose([current_pos.x_val, current_pos.y_val, current_pos.z_val], wp, atol=1.5):
            current_coords = np.array([current_pos.x_val, current_pos.y_val, current_pos.z_val])
            drone_path.append([current_pos.x_val, current_pos.y_val, current_pos.z_val])
            
            control_signal = pidC.update(current_coords, dt=1)
            control_signal = np.clip(control_signal, -5, 5)
            
            yaw_angle = calculate_yaw_angle(current_coords, wp)
            client.moveByVelocityAsync(
                control_signal[0]/2, 
                control_signal[1]/2, 
                control_signal[2]/5, 
                0.5,
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(False, yaw_angle)
            ).join()
            
            collision_info = client.simGetCollisionInfo()
            if collision_info.has_collided:
                collision_count += 1 

            final_approach_velocity = [control_signal[0]/5, control_signal[1]/5, control_signal[2]/5]
            current_pos = client.getMultirotorState().kinematics_estimated.position
            start_time = time.time()
            if(start_time - stuck_timestamp > 10 ):
                client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(wp[0], wp[1], wp[2]), airsim.to_quaternion(0, 0, 0)), True)
                print("fixing stuck")

            print("TIME:" , start_time - run_start_time)
            if(start_time - run_start_time > 120): 
                client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(6.788, 81.6774, -43.380), 
                                    airsim.to_quaternion(0, 0, 0)), True)
                return np.array(gate_clearance_positions), collision_count, start_time - run_start_time

        if i < len(WAYPOINTS) - 1:
            print("Clearing gate...")
            # Store position at gate clearance
            current_pos = client.getMultirotorState().kinematics_estimated.position
            gate_clearance_positions.append([current_pos.x_val, current_pos.y_val, current_pos.z_val])
            
            clearance_time = 1.5
            # if(i == 12 ): clearance_time =4
            start_time = time.time()
            # print("TIME:" , start_time - run_start_time)
            if(start_time - run_start_time > 240): 
                client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(6.788, 81.6774, -43.380), 
                                    airsim.to_quaternion(0, 0, 0)), True)
                return np.array(gate_clearance_positions), collision_count, start_time - run_start_time

            while time.time() - start_time < clearance_time:
                current_pos = client.getMultirotorState().kinematics_estimated.position
                drone_path.append([current_pos.x_val, current_pos.y_val, current_pos.z_val])
                
                client.moveByVelocityAsync(
                    final_approach_velocity[0],
                    final_approach_velocity[1],
                    final_approach_velocity[2],
                    0.1,
                    airsim.DrivetrainType.MaxDegreeOfFreedom,
                    airsim.YawMode(False, yaw_angle)
                ).join()

                collision_info = client.simGetCollisionInfo()
                if collision_info.has_collided:
                    collision_count += 1

    end_time = time.time()
    total_runtime = end_time - run_start_time
    print("Completed all waypoints")
    return np.array(gate_clearance_positions), collision_count, total_runtime


def get_sim_picture():
    responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])
    img_rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height, responses[0].width, 3)
    return img_rgb

def get_coords_vision(image):
    # This is where some Agent processes the image, and returns me data @robert looking into what it will return, and how i can use it
    return WAYPOINTS[0]  # Replace this with real processing

def vision_based_navigation():
    for _ in range(len(WAYPOINTS)):  # Loop through as many gates as there are waypoints
        client.simPause(True)
        image = get_sim_picture()
        coords = get_coords_vision(image)
        print(f"Vision-based coords: {coords}")
        
        # Resume simulation and navigate to new coordinates
        client.simPause(False)
        client.moveByVelocityAsync(coords[0], coords[1], coords[2], 5).join()


if __name__ == "__main__":
    

    gain_configurations = [
    {'gain_x': [7, 0, 8.0], 'gain_y': [7, 0, 8.0], 'gain_z': [6, 0, 5.0]},
    {'gain_x': [3, 0, 9.0], 'gain_y': [3, 0, 9.0], 'gain_z': [6, 0, 5.0]},
    {'gain_x': [4, 0, 8.5], 'gain_y': [4, 0, 8.5], 'gain_z': [6, 0, 5.0]},
    {'gain_x': [5, 0, 8.0], 'gain_y': [5, 0, 8.0], 'gain_z': [6, 0, 5.0]}

    ]

    target_x=0
    target_y=0
    target_z =2.0199999809265137
    # Run each configuration
    for gains in gain_configurations:
        print(f"\nTesting gains: {gains}")
        
        # Reset drone position
        client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(target_x, target_y, target_z), 
                                            airsim.to_quaternion(0, 0, 0)), True)


        time.sleep(3)
        client.takeoffAsync().join()
        
        # Initialize PID controller with current gains
        pidC = PIDController(gain_x=gains['gain_x'], 
                          gain_y=gains['gain_y'], 
                          gain_z=gains['gain_z'])
        
        # Run controller and store results
        gate_positions, collision_count, total_runtime  = state_based_pid_control(pidC)

        client.landAsync().join()
        client.armDisarm(True)
        client.enableApiControl(True)
        time.sleep(5)  # Wait between runs


