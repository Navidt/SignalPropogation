import csv

import matplotlib.pyplot as plt

with open('rssi.csv', 'r') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    next(data, None)  # skip the headers

    x = []
    y = []

    for row in data:
        x.append(float(row[0]))
        y.append(int(row[3]))

    min_x = min(x)
    x = [value - min_x for value in x]

    plt.plot(x, y)
    plt.title('RSSI vs Time')  # Title
    plt.xlabel('Time')  # X-axis label
    plt.ylabel('RSSI (dBm)')  # Y-axis label
    plt.legend()
    plt.show()
