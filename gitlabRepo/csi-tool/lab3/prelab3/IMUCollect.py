from sense_hat import SenseHat
import numpy as np
import time
import scipy.signal as signal
from datetime import datetime,date
import matplotlib.pyplot as plt
import pandas as pd

import scipy.integrate as integrate
# path="/home/pi/lab3/" # Change the path to your current folder (you can use `pwd` command in terminal to find the full path)
path = "/home/pi/.config/sense_hat/"  
sense=SenseHat()

timestamp_fname=datetime.now().strftime("%H:%M:%S")
sense.set_imu_config(True,True,True) ## Config the Gyroscope, Accelerometer, Magnetometer
filename=path+timestamp_fname+".csv"

with open(filename,"w") as f:
    print("collecting data now")
    while True:
        accel=sense.get_accelerometer_raw()  ## returns float values representing acceleration intensity in Gs
        orient=sense.get_orientation_radians()  ## returns float values representing rotation of the axis in radians
        mag=sense.get_compass_raw()  ## returns float values representing magnetic intensity of the ais in microTeslas
        gyro=sense.get_gyroscope_raw()
    
        x=accel['x']
        y=accel['y']
        z=accel['z']
        pitch = orient['pitch']
        roll = orient['roll']
        yaw = orient['yaw']
        gyro_x=gyro['x']
        gyro_y=gyro['y']
        gyro_z=gyro['z']
        timestamp=datetime.now().strftime("%H:%M:%S")
        entry= str(time.time())+","+timestamp+","+str(x)+","+str(y)+","+str(z)+","+ str(pitch)+ ","+str(roll)+","+ str(yaw)+ ","+ str(mag['x'])+ ","+str(mag['y'])+","+ str(mag['z'])+","+str(gyro_x)+","+str(gyro_y)+","+str(gyro_z)+"\n"
    
    

        
        f.write(entry)
   
        
f.close()
        


