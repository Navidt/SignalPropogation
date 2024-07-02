import numpy as np
import time
from datetime import datetime,date
import matplotlib.pyplot as plt
import pandas as pd


## Install the following libraries if they are not already installed on your Raspi:
## pip3 install seaborn
## sudo apt-get install python3-pandas
## sudo apt-get install python3-matplotlib



filename="/home/pi/.config/sense_hat/17:06:03.csv" #step2
# filename="/home/pi/.config/sense_hat/17:17:46.csv" #step1

## CSV file template:
# time in seconds, timestamp (H:M:S), X-Accel, Y-Accel, Z-Accel, pitch, roll, yaw, X-Mag, Y-Mag, Z-Mag, X-Gyro, Y-Gyro, Z-Gyro


df =pd.read_csv(filename, header=None)
df=df.dropna()

timestamp = df[0]

timestamps2=df[1]
x_axis=df[2]
y_axis=df[3]
z_axis=df[4]
plt.plot(x_axis,  label="X-axis Raw Acceleration")
plt.plot(y_axis,  label="Y-axis Raw Acceleration")
plt.plot(z_axis,  label="Z-axis Raw Acceleration")
plt.legend(loc="upper left")
plt.ylabel("Raw Acceleration in m/s^2")
plt.xlabel("Number of Data Points")
plt.show()

### CALIBERATION
x_calib_mean = np.mean(x_axis[1:100])
## caliberate x,y,z to reduce the bias in accelerometer readings. Subtracting it from the mean means that in the absence of motion, the accelerometer reading is centered around zero to reduce the effect of integrigation drift or error.
## change the upper and lower bounds for computing the mean where the RPi is in static position at the begining of the experiment (i.e. for the first few readings). You can know these bounds from the exploratory plots above.
x_calib = x_axis - x_calib_mean
x_calib = x_calib[:]
timestamp = timestamp[:]

y_calib_mean = np.mean(y_axis[1:100])
y_calib = y_axis - y_calib_mean
y_calib = y_calib[:]
timestamp = timestamp[:]

z_calib_mean = np.mean(y_axis[1:100])
z_calib = z_axis - z_calib_mean
z_calib = z_calib[:]
timestamp = timestamp[:]

plt.plot(x_calib, label="X-axis Caliberated Acceleration")
plt.plot(y_calib, label="Y-axis Caliberated Acceleration")
plt.plot(z_axis, label="Z-axis Caliberated Acceleration")
plt.legend(loc="upper left")
plt.ylabel("Caliberated Acceleration in m/s^2")
plt.xlabel("Number of Data Points")
plt.show()


#print("Check if lengths of each vector are same for tracking time", len(timestamp), len(x_calib), len(y_calib), len(z_calib))
# Find sampling time:
dt = (timestamp[len(timestamp)-1] - timestamp[0]) / len(timestamp)

## Computing Velocity and Position along Y Axis:
y_vel = [0]
for i in range(len(y_calib)-1):
    y_vel.append(y_vel[-1] + dt * y_calib[i])

y = [0]

for i in range(len(y_vel)-1):
    y.append(y[-1] + dt * y_vel[i])
    
## Integrations along X axis
x_vel = [0]
for i in range(len(x_calib)-1):
    x_vel.append(x_vel[-1] + dt * x_calib[i])

plt.plot(x_vel, label="X-axis velocity")
plt.plot(y_vel, label="Y-axis velocity")
plt.legend(loc="upper left")
plt.ylabel("Velocity in m/s")
plt.xlabel("# of Samples")
plt.show()

x = [0]

for i in range(len(x_vel)-1):
    x.append(x[-1] + dt * x_vel[i])
    
## Integrations along Z axis:

z_vel = [0]
for i in range(len(z_calib)-1):
    z_vel.append(z_vel[-1] + dt * z_calib[i])

#plt.plot(z_vel)

z = [0]
plt.figure()
for i in range(len(z_vel)-1):
    z.append(z[-1] + dt * z_vel[i])
#plt.plot(z)

## Plot X and Y positions with respect to time:
plt.plot(timestamp,x, label="X positions", c="red")
plt.plot(timestamp,y, label="Y positions", c="green")
plt.legend(loc="upper left")
plt.xlabel("Timestamp (seconds)")
plt.ylabel("Positions (m)")
plt.show()


## Visualizing scatter plot in 2D. Fill in if you want to see this data

##

## Orientation Integration

# yaw = df[7]
# plt.plot(yaw)
# plt.show()

# x_vel_corrected = [0]
# y_vel_corrected = [0]
# x_dir_corrected = [0]
# y_dir_corrected = [0]

# for i in range(len(x_calib)-1):
#     x_vel_corrected.append(x_vel[-1]*np.cos(yaw[i]) + dt * x_calib[i])
    
# for i in range(len(y_calib)-1):
#     y_vel_corrected.append(y_vel[-1]*np.sin(yaw[i]) + dt * y_calib[i])
    
# for i in range(len(x_vel_corrected)-1):
#     x_dir_corrected.append(x_dir_corrected[-1] + dt * x_vel_corrected[i])

# for i in range(len(y_vel_corrected)-1):
#     y_dir_corrected.append(y_dir_corrected[-1] + dt * y_vel_corrected[i])
    
# plt.plot(timestamp,x, label="X positions", c="red")
# plt.plot(timestamp,y, label="Y positions", c="green")
# plt.plot(timestamp,x_dir_corrected, label="X positions Corrected")
# plt.plot(timestamp,y_dir_corrected, label="Y positions Corrected")
# plt.legend(loc="upper left")
# plt.xlabel("Timestamp (seconds)")
# plt.ylabel("Positions (m)")
# plt.show()