import numpy as np

class PIDController:
    def __init__(self, gain_x, gain_y, gain_z, setpoint=[0, 0, 0], output_limit=10):
        self.Kp_x, self.Ki_x, self.Kd_x = gain_x
        self.Kp_y, self.Ki_y, self.Kd_y = gain_y
        self.Kp_z, self.Ki_z, self.Kd_z = gain_z

        self.Kp = np.array([self.Kp_x, self.Kp_y, self.Kp_z])
        self.Ki = np.array([self.Ki_x, self.Ki_y, self.Ki_z])
        self.Kd = np.array([self.Kd_x, self.Kd_y, self.Kd_z])

        self.setpoint = np.array(setpoint)
        self.prev_error = np.zeros(3)
        self.integral = np.zeros(3)
        self.output_limit = output_limit
        
        # Add deadband to reduce oscillations near target
        self.deadband = 0.3
        # Add separate limits for position-based gain reduction
        self.position_threshold = 2.0

    def update(self, current_value, dt):
        error = self.setpoint - current_value
        error_magnitude = np.linalg.norm(error)
        
        # Apply deadband to reduce small oscillations
        error = np.where(np.abs(error) < self.deadband, 0, error)
        
        # Reduce gains when close to target to prevent overshooting
        if error_magnitude < self.position_threshold:
            position_scale = error_magnitude / self.position_threshold
            current_p = self.Kp * position_scale
            current_d = self.Kd * position_scale
        else:
            current_p = self.Kp
            current_d = self.Kd
            
        # self.integral += error * dt
        
        # More aggressive integral windup prevention
        # integral_limit = 5  # Reduced from 10
        self.integral = 0 #np.clip(self.integral, -integral_limit, integral_limit)
        
        derivative = (error - self.prev_error) / max(dt, 0.01)
        
        # Calculate control signal with scaled gains
        output = current_p * error + self.Ki * 0 + current_d * derivative
        
        # More conservative output limiting
        output = np.clip(output, -self.output_limit, self.output_limit)
        
        self.prev_error = error
        return output

    def update_setpoint(self, setpoint):
        self.setpoint = np.array(setpoint)
        # Reset integral term when changing setpoints to prevent accumulation
        self.integral = np.zeros(3)



import airsim
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from particle_main import RunParticle
import traceback
import random
# from controller_m.gen_traj import Generate
# from perception.perception import Perception
# from simple_excitation import excitation

from pyvista_visualiser import Perception_simulation
import matplotlib.animation as animation



# def calculate_performance_metrics(position_history, settlingpoint, tolerance_percentage=2):
#     pos_adj = position_history - position_history[0]
#     peak = np.max(pos_adj)
#     Mp = 100* (peak - settlingpoint) / settlingpoint   # Overshoot as percentage
#     # Find time index where percent difference between current pos and settling point is less than predetermined tolerance 
#     Ts_idx = np.where(np.abs(pos_adj - settlingpoint) <= tolerance_percentage / 100 * settlingpoint)[0][0]
#     settling_time = Ts_idx * timestep  # Using Predefined tiemstep
#     #Rise time
#     Tr_idx = np.where(pos_adj >= 0.9 * settlingpoint)[0]
#     rise_time = Tr_idx[0] * timestep  
#     return Mp, settling_time, rise_time



def calculate_performance_metrics(position_history, settlingpoint, tolerance_percentage=2, timestep=None):
    pos_adj = position_history - position_history[0]  # Adjust by init pos
    settling_time=0
    rise_time=0
    # overshoot
    peak = np.max(pos_adj)
    Mp = 100 * (peak - settlingpoint) / settlingpoint  # Mp percentage

    # Check if timestep is provided for time-based metrics
    if timestep > -np.inf:
    # settling time
        tolerance = tolerance_percentage / 100 * settlingpoint
        Ts_idx = np.where(np.abs(pos_adj - settlingpoint) <= tolerance)[0]
        settling_time = Ts_idx[0] * timestep  # First time where pos is within tolerance

    # rise time
        Tr_idx = np.where(pos_adj >= 0.9 * settlingpoint)[0]
        rise_time = Tr_idx[0] * timestep  # First time where 90% of settling point is reached
    return Mp, settling_time, rise_time



lead = "Drone_C"
if __name__ == "__main__":


    # connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()
    

    client.enableApiControl(True,lead)
    client.armDisarm(True, lead)
    client.takeoffAsync(20.0, lead).join()

    total = client.getMultirotorState(lead).kinematics_estimated
    current_pos = [client.getMultirotorState(lead).kinematics_estimated.position.x_val, client.getMultirotorState(lead).kinematics_estimated.position.y_val, client.getMultirotorState(lead).kinematics_estimated.position.z_val]
    print("kinematics",total)

 
    count = 0
    totalcount = 1000
    timestep = 0.01  # Time step
    dt = 0.1
    position_history = []

    gain_x = [20,0,100,0]
    gain_y = [20,0,100.0]
    gain_z = [2,0,20.0]
    setpoint = [1,1,35]
    pid_controller = PIDController(gain_x=gain_x, gain_y=gain_y, gain_z=gain_z, setpoint=setpoint)
    
    # Initialize PID controller with desired gains and setpoint
    print("current pos",current_pos)
    world = [client.simGetObjectPose(lead).position.x_val,client.simGetObjectPose(lead).position.y_val,client.simGetObjectPose(lead).position.z_val] 
    print("world pose, ",world  )

    step_target = [current_pos[0]+5,current_pos[1]+5,current_pos[2]+5]
    # step_target = [10,10,10]
    print("step target ",step_target)


    try:
        while True:
            client = airsim.MultirotorClient()
            client.confirmConnection()
            current_velocity = [client.getMultirotorState(lead).kinematics_estimated.linear_velocity.x_val, client.getMultirotorState(lead).kinematics_estimated.linear_velocity.y_val,client.getMultirotorState(lead).kinematics_estimated.linear_velocity.z_val]
            
            current_pos = np.array([client.getMultirotorState(lead).kinematics_estimated.position.x_val, client.getMultirotorState(lead).kinematics_estimated.position.y_val, client.getMultirotorState(lead).kinematics_estimated.position.z_val])
    
            # Compute control signal using PID controller
            control_signal = pid_controller.update(current_pos, dt)
            
            # Update quadrotor velocity using control signal
            current_velocity[0] += control_signal[0] * dt
            current_velocity[1] += control_signal[1] * dt
            current_velocity[2] += control_signal[2] * dt
        
            # client.moveByVelocityZBodyFrameAsync(current_velocity[0],current_velocity[1],10, timestep, vehicle_name = lead)
            client.moveByVelocityAsync(current_velocity[0],current_velocity[1],current_velocity[2], timestep, vehicle_name = lead)
            

            count += 1

            time.sleep(timestep)

            if count == totalcount:
                break

            position_history.append(current_pos)
        

        client.reset()
        client.armDisarm(False)
        client.enableApiControl(False)

        position_history=np.array(position_history)
        times = np.arange(0,position_history.shape[0])*timestep
        fig, (posx,posy,posz) = plt.subplots(3, 1, figsize=(14, 10))

        posx.plot(times, position_history[:,0], label = "Pos x")
        posx.legend()
        posy.plot(times, position_history[:,1], label = "Pos y")
        posy.legend()
        posz.plot(times, position_history[:,2], label = "Pos z")    
        posy.legend()

        plt.show()



    except Exception as e:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("Error Occured, Canceling: ",e)
        traceback.print_exc()

        client.reset()
        client.armDisarm(False)

        # that's enough fun for now. let's quit cleanly
        client.enableApiControl(False)