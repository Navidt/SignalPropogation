import csv
import numpy as np
import matplotlib.pyplot as plt

rssi = []
x = []
y = []

with open('rssi.csv', 'r') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    next(data, None)  # skip the headers

    for row in data:
        x.append(float(row[4]))
        y.append(float(row[5]))
        rssi.append(int(float(row[3])))

plt.scatter(x, y, c=rssi, cmap='viridis')  # 'c' maps the values to colors
plt.colorbar()
plt.show()