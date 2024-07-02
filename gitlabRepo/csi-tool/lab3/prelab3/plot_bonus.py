import csv

import matplotlib.pyplot as plt
import numpy as np

rssi = []
x = []
y = []

rssi_interpolated = []
x_interpolated = []
y_interpolated = []

with open('rssi.csv', 'r') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    next(data, None)  # skip the headers

    for row in data:
        rssi.append(float(row[3]))
        x.append(float(row[4]))
        y.append(float(row[5]))

        if len(x) >= 2 and len(y) >= 2:
            # Vertical line interpolation
            if x[-2] == x[-1] and y[-2] != y[-1]:
                y_new = np.linspace(min(y[-2], y[-1]), max(y[-2], y[-1]), 100)
                rssi_new = np.interp(y_new, [y[-2], y[-1]], [rssi[-2], rssi[-1]])

                x_interpolated.extend([x[-1]] * 100)
                y_interpolated.extend(y_new.tolist())
                rssi_interpolated.extend(rssi_new.tolist())

            # Horizontal line interpolation
            elif y[-2] == y[-1] and x[-2] != x[-1]:
                x_new = np.linspace(min(x[-2], x[-1]), max(x[-2], x[-1]), 100)
                rssi_new = np.interp(x_new, [x[-2], x[-1]], [rssi[-2], rssi[-1]])

                x_interpolated.extend(x_new.tolist())
                y_interpolated.extend([y[-1]] * 100)
                rssi_interpolated.extend(rssi_new.tolist())

# Combining original and interpolated points before plotting
x.extend(x_interpolated)
y.extend(y_interpolated)
rssi.extend(rssi_interpolated)

plt.scatter(x, y, c=rssi, cmap='viridis')
plt.colorbar()
plt.show()