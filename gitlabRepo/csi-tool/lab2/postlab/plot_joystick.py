import csv
import numpy as np
import matplotlib.pyplot as plt

rtime = []
rssi = []

with open('rssi.csv', 'r') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    next(data, None)  # skip the headers

    for row in data:
        rtime.append(float(row[0]))
        rssi.append(int(float(row[3])))

jtime = []
x = []
y = []

with open('joystick.csv', 'r') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    next(data, None)  # skip the headers


    for row in data:
        jtime.append(float(row[0]))
        x.append(int(row[2]))
        y.append(int(row[3]))

xplt = []
yplt = []

for t in rtime:
    idx = np.searchsorted(jtime, t, side='left')
    xplt.append(x[idx])
    yplt.append(y[idx])

plt.scatter(xplt, yplt, c=rssi, cmap='viridis')  # 'c' maps the values to colors
plt.colorbar()
plt.show()