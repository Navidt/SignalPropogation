import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate as it

filename = "04-06-15:14.csv"

df = pd.read_csv(filename, header=None)
df.columns = df.iloc[0]
df = df[1:]
df = df.reset_index(drop=True)
df.dropna()

timestamps = df["timestamp"].astype(np.float32)
x_axis = df["x"].astype(np.float32)
y_axis = df["y"].astype(np.float32)
z_axis = df["z"].astype(np.float32)
rssi_df = df["rssi"].astype(np.float32)


dt = timestamps.diff()
print("-------dt-------")
print(dt)

# rssi value are NaN in the first few rows
first_rssi_index = df["rssi"].first_valid_index()

'''
CALIBRATION
Calibrate x,y,z to reduce the bias in accelerometer readings. 
Subtracting it from the mean means that in the absence of motion, the accelerometer reading is centered around zero.
Change the upper and lower bounds for computing the mean when in static position at the beginning of the experiment
(i.e. for the first few readings). You can know these bounds from the exploratory plots above.
'''

x_calib_mean = np.mean(x_axis[1:70])
x_calib = round(x_axis - x_calib_mean, 4)

y_calib_mean = np.mean(y_axis[1:70])
y_calib = round(y_axis - y_calib_mean, 4)

z_calib_mean = np.mean(y_axis[1:100])
z_calib = z_axis - z_calib_mean

dt_mean = np.mean(dt.dropna())


print("-------x_calib-------")
print(x_calib)



# Define system matrices
A = np.array([[1, dt_mean, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, dt_mean],
              [0, 0, 0, 1]])

H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

# Define initial state and covariance
x_init = np.zeros((4, 1))  # Initial state estimate
P_init = np.diag([0.1, 0.01, 0.1, 0.01])  # Initial error covariance
# P_init = np.diag([0.5, 0.7, 0.5, 0.7])  # Initial error covariance

# Process noise covariance
Q = np.diag([0.01, 0.01, 0.01, 0.01])

# Measurement noise covariance
R = np.diag([0.1, 0.1])

# Initialize Kalman filter
x_kalman = x_init
P_kalman = P_init

# List to store estimated positions
estimated_positions = []
x_accumulated = [0]
y_accumulated = [0]

# Kalman filter loop
for i in range(len(x_calib)):
    # Prediction
    x_pred = A @ x_kalman
    P_pred = A @ P_kalman @ A.T + Q

    # Correction (update)
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    z = np.array([[x_calib[i]], [y_calib[i]]])
    x_kalman = x_pred + K @ (z - H @ x_pred)
    P_kalman = (np.eye(4) - K @ H) @ P_pred

    # Save estimated position
    x_accumulated.append(x_accumulated[-1] + x_kalman[1, 0])
    y_accumulated.append(y_accumulated[-1] + x_kalman[3, 0])
    # estimated_positions.append((x_kalman[0, 0], x_kalman[2, 0]))


# Extract x and y estimated positions
print("-------x_pos-------")
x_pos = pd.DataFrame(x_accumulated)
print(x_pos)
print("-------y_pos-------")
y_pos = pd.DataFrame(y_accumulated)
print(y_pos)

print(rssi_df)
# rssi_df


window_size = 3
 
def moving_average(input_list, window_size):
    output_list = []
    padding = window_size // 2
    
    # Pad the input list to handle edges
    padded_list = [input_list[0]] * padding + input_list + [input_list[-1]] * padding
    
    for i in range(len(input_list)):
        # Calculate the average of elements within the window
        average = sum(padded_list[i:i + window_size]) / window_size
        output_list.append(average)
        
    return output_list

x_accumulated = moving_average(x_accumulated, 25)
y_accumulated = moving_average(y_accumulated, 25)


rssi = []
x = []
y = []
rssi_df = rssi_df.values.tolist()
rssi_df.append(rssi_df[-1])
# print(rssi_df)

rssi_interpolated = []
x_interpolated = []
y_interpolated = []

for i in range(len(x_pos)):
    rssi.append(rssi_df[i])
    x.append(x_accumulated[i])
    y.append(y_accumulated[i])

    if len(x) >= 2 and len(y) >= 2:
        # Vertical line interpolation
        if x[-2] == x[-1] and y[-2] != y[-1]:
            y_new = np.linspace(min(y[-2], y[-1]), max(y[-2], y[-1]), 200)
            rssi_new = np.interp(y_new, [y[-2], y[-1]], [rssi[-2], rssi[-1]])

            x_interpolated.extend([x[-1]] * 200)
            y_interpolated.extend(y_new.tolist())
            rssi_interpolated.extend(rssi_new.tolist())

        # Horizontal line interpolation
        elif y[-2] == y[-1] and x[-2] != x[-1]:
            x_new = np.linspace(min(x[-2], x[-1]), max(x[-2], x[-1]), 200)
            rssi_new = np.interp(x_new, [x[-2], x[-1]], [rssi[-2], rssi[-1]])

            x_interpolated.extend(x_new.tolist())
            y_interpolated.extend([y[-1]] * 200)
            rssi_interpolated.extend(rssi_new.tolist())

# Combining original and interpolated points before plotting
x.extend(x_interpolated)
y.extend(y_interpolated)
rssi.extend(rssi_interpolated)

plt.scatter(x, y, c=rssi, cmap='viridis')
plt.colorbar()
plt.savefig('scatter_plot.png')
plt.close()
plt.show()


















# plt.plot(x_calib, label="X-axis Caliberated Acceleration")
# plt.plot(y_calib, label="Y-axis Caliberated Acceleration")
# plt.plot(z_axis, label="Z-axis Caliberated Acceleration")
# plt.legend(loc="upper left", fontsize="10")
# plt.ylabel("Caliberated Acceleration in m/s^2")
# plt.xlabel("Number of Data Points")
# plt.show()
